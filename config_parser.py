import argparse
from email.policy import default


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',
                        type=str,
                        choices=['train', 'test'],
                        default='train')
    parser.add_argument('--train_1', required=True, help='path to dataset')
    parser.add_argument('--train_2', required=False, help='path to dataset')
    parser.add_argument('--unl_train_1',
                        required=False,
                        help="path to unlabeled dataset")
    parser.add_argument('--unl_train_2',
                        required=False,
                        help="path to unlabeled dataset")
    parser.add_argument('--unl_train_3',
                        required=False,
                        help="path to unlabeled dataset")
    parser.add_argument('--unl_train_4',
                        required=False,
                        help="path to unlabeled dataset")
    parser.add_argument('--unl_train_5',
                        required=False,
                        help="path to unlabeled dataset")
    parser.add_argument('--unl_train_6',
                        required=False,
                        help="path to unlabeled dataset")
    parser.add_argument('--unl_train_7',
                        required=False,
                        help="path to unlabeled dataset")
    parser.add_argument('--batchSize',
                        type=int,
                        default=384,
                        help='input batch size')
    parser.add_argument('--unl_batchSize',
                        type=int,
                        default=288,
                        help='input batch size')
    parser.add_argument('--eval_data',
                        default='eval_and_val/evaluation/',
                        help='path to validation dataset')
    parser.add_argument('--valid_data',
                        default='eval_and_val/validation/',
                        help='path to validation dataset')
    parser.add_argument('--workers',
                        type=int,
                        default=4,
                        help='number of data loading workers')
    parser.add_argument('--unl_workers',
                        type=int,
                        default=4,
                        help='number of data loading workers')
    parser.add_argument('--num_iter',
                        type=int,
                        default=250000,
                        help='number of iterations to train for')
    parser.add_argument('--val_interval',
                        type=int,
                        default=2000,
                        help='Interval between each validation')
    parser.add_argument('--grad_clip',
                        type=float,
                        default=5,
                        help='gradient clipping value. default=5')
    parser.add_argument('--data_usage_ratio',
                        type=float,
                        default=1,
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--unlabel_data_usage_ratio',
                        type=float,
                        default=1,
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--robust',
                        default=False,
                        # action='store_true',
                        type=bool,
                        help='For robust smoothing or not')
    parser.add_argument('--calibrator',
                    type=str,
                    default='CE',
                    help='calibrator')
    parser.add_argument('--alpha',
                    type=float,
                    default=0.05,
                    help='calibrate smoothing strength')
    parser.add_argument('--exp_base',
                    type=int,
                    default=4,
                    help='global trust exp')
    parser.add_argument('--transit_time_ratio',
                    type=float,
                    default=0.2,
                    help='start to calibrate')
    parser.add_argument('--use_ada_threshold',
                        default=False,
                        type=bool,
                        help='For adaptive_threshold_generate or not')
    parser.add_argument('--language', type=str, default='en')
    """ Optimizer """
    parser.add_argument('--optimizer',
                        type=str,
                        default='adam',
                        help='optimizer |sgd|adadelta|adam|')
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='learning rate, default=1.0 for Adadelta, 0.0005 for Adam')
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument(
        '--schedule',
        default='super',
        nargs='*',
        help=
        '(learning rate schedule. default is super for super convergence, 1 for None, [0.6, 0.8] for the same setting with ASTER'
    )
    parser.add_argument('--model_name',
                        type=str,
                        required=True,
                        help='CRNN|TRBA|TRBC')
    parser.add_argument('--num_fiducial',
                        type=int,
                        default=20,
                        help='number of fiducial points of TPS-STN')
    parser.add_argument(
        '--input_channel',
        type=int,
        default=3,
        help='the number of input channel of Feature extractor')
    parser.add_argument(
        '--output_channel',
        type=int,
        default=512,
        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size',
                        type=int,
                        default=256,
                        help='the size of the LSTM hidden state')
    parser.add_argument('--batch_max_length',
                        type=int,
                        default=25,
                        help='maximum-label-length')
    parser.add_argument('--imgH',
                        type=int,
                        default=32,
                        help='the height of the input image')
    parser.add_argument('--imgW',
                        type=int,
                        default=100,
                        help='the width of the input image')
    parser.add_argument(
        '--character',
        type=str,
        default=
        # '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~',
        '0123456789abcdefghijklmnopqrstuvwxyz',
        help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--NED',
                        action='store_true',
                        help='For Normalized edit_distance')
    parser.add_argument('--Aug',
                        type=str,
                        default='rand',
                        choices=['rand', 'None', 'weak'])
    parser.add_argument('--exp_name', help='Where to store logs and models')
    parser.add_argument('--manual_seed',
                        type=int,
                        default=111,
                        help='for random seed setting')
    parser.add_argument('--saved_model',
                        default='',
                        help="path to model to continue training")
    parser.add_argument('--displayInterval', default=5000, type=int)
    parser.add_argument('--checkpoint_root', default="saved_models", type=str)
    parser.add_argument('--json_path', type=str, default='None')
    """ Semi-supervised learning """
    parser.add_argument(
        '--semi',
        type=str,
        default='None',
        help='whether to use semi-supervised learning |None|KLDiv|CrossEntropy|PseudoLabel|'
    )
    parser.add_argument('--Aug_semi',
                        type=str,
                        default='rand',
                        choices=['rand', 'None', 'weak'])
    parser.add_argument('--ema_alpha',
                        type=float,
                        default=0.999,
                        help='EMA decay')

    # for semi supervised
    parser.add_argument('--lambda_cons',
                        type=float,
                        default=1,
                        help='Mean Teacher consistency weight')
    parser.add_argument('--lambda_mmd', default=0.01, type=float)
    parser.add_argument('--confident_threshold', default=0.5, type=float)
    parser.add_argument('--l_confident_threshold', default=0.6, type=float)
    parser.add_argument('--uda_softmax_temp', default=0.4, type=float)
    parser.add_argument('--eval_type', default="simple")
    parser.add_argument('--projection_type',
                        type=str,
                        choices=['pff', 'linear'],
                        default='pff')

    opt = parser.parse_args()
    if opt.model_name == 'CRNN':  # CRNN = NVBC
        opt.Transformation = 'None'
        opt.FeatureExtraction = 'VGG'
        opt.SequenceModeling = 'BiLSTM'
        opt.Prediction = 'CTC'

    elif opt.model_name == 'TRBC':  # TRBC
        opt.Transformation = 'TPS'
        opt.FeatureExtraction = 'ResNet'
        opt.SequenceModeling = 'BiLSTM'
        opt.Prediction = 'CTC'

    elif opt.model_name == 'TRBA':  # TRBA
        opt.Transformation = 'TPS'
        opt.FeatureExtraction = 'ResNet'
        opt.SequenceModeling = 'BiLSTM'
        opt.Prediction = 'Attn'

    elif opt.model_name == 'RBA':  # RBA
        opt.Transformation = 'None'
        opt.FeatureExtraction = 'ResNet'
        opt.SequenceModeling = 'BiLSTM'
        opt.Prediction = 'Attn'
    opt.run_code_root = "saved_models"
    return opt