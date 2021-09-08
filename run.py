import argparse
import logging
import sys

from pup.config import Config
from pup.experiment import exp_reconstruction_training, exp_reconstruction_predict, \
    exp_quantify_voi

logger = logging.getLogger(__name__)


TASK_BENCHMARK = 'benchmark'
TASKS = [TASK_BENCHMARK]

COMPONENT_RECONSTRUCTION_TRAINING = 'reconstruction_training'
COMPONENT_RECONSTRUCTION_PREDICT = 'reconstruction_predict'
COMPONENT_QUANTIFY_VOI = 'quantify_voi'
COMPONENTS = [
    COMPONENT_RECONSTRUCTION_TRAINING,
    COMPONENT_RECONSTRUCTION_PREDICT,
    COMPONENT_QUANTIFY_VOI
]


class MyParser(argparse.ArgumentParser):
    """ An parse to print help whenever an error occurred to the parsing process
    """
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)


if __name__ == '__main__':
    parser = MyParser(description='Execute a task')
    parser.add_argument('-t',
                        '--task',
                        help='Task name',
                        choices=TASKS,
                        required=True)

    parser.add_argument('-c',
                        '--component',
                        help='Component name',
                        choices=COMPONENTS,
                        required=False)

    parser.add_argument('-v',
                        '--verbose',
                        help='increase output verbosity',
                        action='store_true')

    parser.add_argument("--eval-only",
                        default=False,
                        help="Flag to only run evaluation when we already have output costs and distributions",
                        action="store_true")

    args = parser.parse_args()

    logger.info('Started')
    logger.info(Config.get_config_str())

    if args.task == TASK_BENCHMARK:
        eval_only = args.eval_only

        if args.component == COMPONENT_RECONSTRUCTION_TRAINING:
            exp_reconstruction_training.exe_exp_reconstruction_training_multi_users()

        if args.component == COMPONENT_RECONSTRUCTION_PREDICT:
            exp_reconstruction_predict.exe_reconstruction_predict_multi_users()

        if args.component == COMPONENT_QUANTIFY_VOI:
            exp_quantify_voi.exe_exp_quantify_voi_multi_users()
