import os
import cPickle
import gzip
import re
from collections import defaultdict

from sklearn.ensemble import RandomForestClassifier
from scoring.data_sets_I.models import NearestNeighbourModel

import configuration_file_I as conf

# TODO make datasetI/II just an argument
# TODO maybe include this here, but this takes some restructuring!
# import configuration_file_II
from scoring import scoring_functions as scor, ml_functions_13 as ml_func
from optparse import OptionParser
from filelock import FileLock

# prepare command-line option parser
cli_parser = OptionParser("usage: %prog [options] arg")
cli_parser.add_option(
    "-n", "--num", dest="num", type="int", metavar="INT", help="number of query mols"
)
cli_parser.add_option(
    "-f", "--fingerprint", dest="fp", help="fingerprint to train ML model with"
)
cli_parser.add_option(
    "-o",
    "--outpath",
    dest="outpath",
    metavar="PATH",
    help="relative output PATH (default: pwd)",
)
cli_parser.add_option(
    "-s",
    "--similarity",
    dest="simil",
    type="string",
    metavar="NAME",
    help="NAME of similarity metric to use (default: Dice, other options are: \
    Tanimoto, Cosine, Russel, Kulczynski, McConnaughey, Manhattan, RogotGoldberg",
)
cli_parser.add_option(
    "-m", "--ml", dest="ml", metavar="FILE", help="file containing the ",
)
cli_parser.add_option(
    "-a",
    "--append",
    dest="do_append",
    action="store_true",
    help="append to the output file (default: False)",
)
cli_parser.add_option(
    "-x",
    "--filter",
    dest="filter_file",
    metavar="FILE",
    help="File containing (dataset, target) tuples describing what targets to use",
)


def get_cwd_from_file_path():
    return os.path.dirname(os.path.realpath(__file__))


CWD = get_cwd_from_file_path()
PARENT_PATH = CWD + "/../../"
INPATH_CMP = PARENT_PATH + "compounds/"
INPATH_LIST = PARENT_PATH + "query_lists/data_sets_I/"


def run_scoring(options, ml_model, path, ml_name_prefix):
    # read in command line options
    # required arguments
    if options.num and options.fp:
        num_query_mols = options.num
        fp_name = options.fp
    else:
        raise RuntimeError("one or more of the required options was not given!")

    # optional arguments
    do_append = False
    if options.do_append:
        do_append = options.do_append
    simil_metric = "Dice"
    if options.simil:
        simil_metric = options.simil
    outpath = path
    outpath_set = False
    if options.outpath:
        outpath_set = True
        outpath = path + options.outpath

    process_target_list = []
    if options.filter_file:
        filter_file = path + options.filter_file
        with open(filter_file, "r") as f:
            for line in f.readlines():
                dataset, target = tuple(
                    map(lambda x: re.sub(r"[\n\t\s]*", "", x), line.split(","))
                )
                # the MUV and ChEMBL targets are ints, not strings!
                try:
                    target = int(target)
                except ValueError:
                    # keep it a string, but sould be DUD
                    pass
                process_target_list.append((dataset, target))
        print "Only processing following targets: {0}".format(process_target_list)

    # check for sensible input
    if outpath_set:
        scor.checkPath(outpath, "output")
    scor.checkSimil(simil_metric)
    scor.checkQueryMols(num_query_mols, conf.list_num_query_mols)

    # initialize machine-learning method

    # FIXME remove debug
    break_now_outer = False
    # loop over data-set sources
    for dataset in conf.set_data.keys():
        print dataset
        for target in conf.set_data[dataset]["ids"]:

            # only process those (dataset, target) combinations
            # that are provided, if there is a list present
            if process_target_list:
                if (dataset, target) not in process_target_list:
                    continue

            print target

            actives_id_smiles = read_molecules_from_file(dataset, target, actives=True)

            num_actives = len(actives_id_smiles)
            num_test_actives = num_actives - num_query_mols
            # convert fps to numpy arrays

            # TODO firstchembl check?
            decoys_id_smiles = read_molecules_from_file(dataset, target, actives=False)
            num_decoys = len(decoys_id_smiles)

            training_list = read_training_list(dataset, num_query_mols, target)

            # test fps and molecule info, exclude the training molecules from the test-set
            #   (-> if i not in training_list ... )
            test_list = [
                i for i in range(num_actives) if i not in training_list[:num_query_mols]
            ]
            test_list += [
                i for i in range(num_decoys) if i not in training_list[num_query_mols:]
            ]

            training_smiles = get_training_smiles(
                actives_id_smiles, decoys_id_smiles, num_query_mols, training_list
            )
            print "Training fingerprints, this can take some time:"
            # XXX This will work with training the fingerprints on the fly, instructing the fiprihash process
            #   over IPC to start the training. Since this is blocking and won't train multiple datasets,
            #   it is best to use a pretrained cache database for the fiprihash process, in order to speedup the
            #   training process significantly
            # Train the fingerprints with the training set, and not the full dataset!

            # the provided can trigger other actions like retrieving multiple
            # fingerprinter with derived parameters from the cache.
            # What FPs have been retrieved from the cache will be determined by
            # the remote fingerprinter.
            # It will return names of fingerprints that are valid and
            # derived from the input name!
            fp_names = scor.trainFP(fp_name, training_smiles, str(target))

            print "Will train following fingerprinter: ", fp_names
            for fp_build in fp_names:

                actives = calc_fingerprints(actives_id_smiles, fp_build)
                decoys = calc_fingerprints(decoys_id_smiles, fp_build)

                print "molecules read in and fingerprints calculated"

                # list with active/inactive info, prepare the target vector, where 'num_query_mols' of actives '1' are
                # present
                ys_fit = [1] * num_query_mols + [0] * (
                    len(training_list) - num_query_mols
                )

                # the pre-calculated fingerprint bitvectors for all the training molecules
                train_fps_actives = [
                    actives[i][1] for i in training_list[:num_query_mols]
                ]
                train_fps_decoys = [
                    decoys[i][1] for i in training_list[num_query_mols:]
                ]

                # the pre-calculated fingerprint bitvectors for all the test molecules
                test_fps = [actives[i][1] for i in test_list[:num_test_actives]]
                test_fps += [decoys[i][1] for i in test_list[num_test_actives:]]

                # the internal ID and active/inactive bool for all the test molecules
                test_mols = [[actives[i][0], 1] for i in test_list[:num_test_actives]]
                test_mols += [[decoys[i][0], 0] for i in test_list[num_test_actives:]]

                # to store the scored lists
                scores = defaultdict(list)
                # loop over repetitions
                for q in range(conf.num_reps):
                    single_score = calculate_single_score(
                        ml_model,
                        ys_fit,
                        train_fps_actives,
                        train_fps_decoys,
                        test_fps,
                        test_mols,
                        simil_metric,
                    )
                    scores[ml_name_prefix + "_" + fp_build].append(single_score)

                # use a filelock, so that no corruption can occur when multiple processes try to append to the file
                # XXX When this crashes before the lock is released, the lock has to be removed manually!
                out_file_path = (
                    outpath + "/list_" + dataset + "_" + str(target) + ".pkl.gz"
                )

                lock = FileLock(out_file_path + ".lock")
                lock.acquire()
                try:
                    if do_append:
                        outfile = gzip.open(out_file_path, "ab+")  # binary format
                    else:
                        outfile = gzip.open(out_file_path, "wb+")  # binary format
                    for fipri_name, score in scores.items():
                        cPickle.dump([fipri_name, score], outfile, 2)
                    outfile.close()
                finally:
                    # FIXME the lockfile is not deleted automatically - this is not a big problem, since
                    #   everything is functional with the lockfiles in place
                    lock.release()

                print "scoring done and scored lists written"


# TODO
#   - in the main function, the pickle dumping is done complying with the format in the calculate_scored_lists_bak.py.
def calculate_single_score(
    ml_model,
    ys_fit,
    train_fps_actives,
    train_fps_decoys,
    test_fps,
    test_mols,
    simil_metric,
):
    add_similarity_to_ml_score = False
    calculate_similarity = False
    train_fps = train_fps_actives + train_fps_decoys
    ml_model.fit(raw_fingerprints_to_numpy(train_fps), ys_fit)
    # rank based on probability
    scores = ml_model.predict_proba(raw_fingerprints_to_numpy(test_fps))

    if isinstance(ml_model, (RandomForestClassifier, NearestNeighbourModel)):
        single_similarities = []
        # calculate similarity with standard fp
        for test_fp, test_mol in zip(test_fps, test_mols):
            # XXX Although we have to calculate active -NN distance score anyways for the RF ranking,
            #   don't save the results but compute them again separately

            #  For NN comparisons (normal and the second sorting score for RF), the comparison is ONLY made
            #  against the ACTIVE molecules from the training list!
            tmp_simil = scor.getBulkSimilarity(test_fp, train_fps_actives, simil_metric)
            tmp_simil.sort(reverse=True)
            # use the highest similarity as measure! ("max fusion")
            # [similarity, internal_id, int(is_active)]
            single_similarities.append([tmp_simil[0], test_mol[0], test_mol[1]])

        # rank based on probability (and second based on similarity)
        # The similarity is only used as a second sorting measure here.
        #  In general, not the values count here, but the rank / active,inactive combination!

        if isinstance(ml_model, NearestNeighbourModel):
            # return the NN scores as the score
            # XXX SORT ON WHOLE ARRAY, use internal id etc as seconday key, since this was done in the
            #   original script, and could change the rank order if done otherwise
            single_score = sorted(single_similarities, reverse=True)
        else:
            # for the RF, add the results of the NN to the ranking
            # store: [probability, similarity, internal ID, active/inactive]
            # XXX SORT ONLY THE FIRST VALUE (COULD OTHERWISE BE SECONDAY SORTED BY INTERNAL_ID),
            #   since this is done in the original script and could change the rank order otherwise
            similarity = [
                elem[0]
                for elem in sorted(
                    single_similarities, reverse=True, key=lambda x: x[0]
                )
            ]
            single_score = [
                [score[1], simil, test_mol[0], test_mol[1]]
                for score, simil, test_mol in zip(scores, similarity, test_mols)
            ]
    else:
        # this is for all other ML methods (LR, NB), don't include the NN in the scores
        # store: [probability, internal ID, active/inactive]
        single_score = [
            [score[1], test_mol[0], test_mol[1]]
            for score, test_mol in zip(scores, test_mols)
        ]
    single_score.sort(reverse=True)
    return single_score


def raw_fingerprints_to_numpy(fingerprints):
    # create expected nested list with None as "dummy" variable
    return ml_func.getNumpy(([None, fp] for fp in fingerprints))


def calc_fingerprints(id_smiles, fp_build):
    fingerprints = []
    for internal_id, smiles in id_smiles:
        fp_dict = scor.getFP(fp_build, smiles)
        # store: [internal ID, dict with fps]
        fingerprints.append([internal_id, fp_dict])
    return fingerprints


def get_training_smiles(
    actives_id_smiles, decoys_id_smiles, num_query_mols, training_list
):
    # gather all smiles for the training molecules:
    training_smiles = [actives_id_smiles[i][1] for i in training_list[:num_query_mols]]
    training_smiles += [decoys_id_smiles[i][1] for i in training_list[num_query_mols:]]
    return training_smiles


def read_training_list(dataset, num_query_mols, target):
    # open training lists
    training_input = open(
        INPATH_LIST
        + dataset
        + "/training_"
        + dataset
        + "_"
        + str(target)
        + "_"
        + str(num_query_mols)
        + ".pkl",
        "r",
    )
    training_list = cPickle.load(training_input)
    return training_list


def read_molecules_from_file(dataset, target, actives):
    if actives is True:
        path = (
            INPATH_CMP
            + dataset
            + "/cmp_list_"
            + dataset
            + "_"
            + str(target)
            + "_actives.dat.gz"
        )
    else:
        if dataset == "ChEMBL":
            path = INPATH_CMP + dataset + "/cmp_list_" + dataset + "_zinc_decoys.dat.gz"
        else:
            path = (
                INPATH_CMP
                + dataset
                + "/cmp_list_"
                + dataset
                + "_"
                + str(target)
                + "_decoys.dat.gz"
            )

    id_smiles = []
    for line in gzip.open(path, "r"):
        if line[0] != "#":
            # structure of line: [external ID, internal ID, SMILES]]
            line = line.rstrip().split()
            id_smiles.append((line[1], line[2]))
    return id_smiles

