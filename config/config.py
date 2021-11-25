import getpass
import configargparse
import time
import os
import json
from datetime import datetime
   

def read_config(file = False):
    if not file:
        parser = configargparse.ArgumentParser(default_config_files = ['config/gemini.cfg'])
    else:
        parser = configargparse.ArgumentParser(default_config_files = [file])
        
    parser.add('-c', '--config_file', is_config_file=True, help='config file path')

    ######################### Operation #######################################################
    parser.add("--extract", action='store_true', help='Run data extraction')
    parser.add("--train", action='store_true', help='Execute model training code')
    parser.add("--predict", action='store_true', help='Run prediction')
    parser.add("--analyze", action='store_true', help='Run analysis')

    ######################### Data Extraction #######################################################

    # database connection parameters
    parser.add("--user", default='koshkinam', type=str, required=False, help='Postgres user')
    parser.add("--password", default=os.environ['PGPASSWORD'], type=str, required=False, help='Postgres password')
    parser.add("--port", default=5432, type=int, help='Postgres port')
    parser.add("--host", default='db.gemini-hpc.ca', type=str, required=False, help='Postgres host')
    parser.add("--database", default='delirium_v3_0_0', type=str, required=False, help='Postgres database')

    # data source and destination parameters
    parser.add('-w', action='store_true', help='Write extracted data to disk')
    parser.add('-r', action='store_true', help='Read from the database')
    parser.add('--input', type=str, default=None, required=False, help='Data file to read from instead of database')
    parser.add("--output_folder", type=str, default='/mnt/nfs/project/delirium/data', help='Which directory should we put the CSV results?')
    parser.add("--output_full_path", type=str, help='Where should we put the CSV results? Full path option.')

    # data extraction parameters
    parser.add('--features', default=[], type= str,  action='append', required=False, help='List of features for the model')
    parser.add('--target', type = str, required = False, help = 'Column we are trying to predict')
    parser.add('--pop_size', type=int, default=10000, required=False, help='Total number of records to read from the database (0 - to read all)')
    parser.add('--filter_year', type=int, default= 0, required = False, help='Select only records from before specified year')

    # specify 'from' and 'to' dates, only records with admit_date in this range will be selected
    parser.add('--filter_date_from', type=str, default='', required = False, help='Format: yyyy-mm-dd. Select starting from this admit_date. Used in conjunction with --filter_date_to')
    parser.add('--filter_date_to', type = str, default='', required = False, help='Format: yyyy-mm-dd. Select before this admit_date. Used in conjunction with --filter_date_from')

    # train/test/val split parameters
    parser.add('--split_column', default='year', type=str, required=False, help='Column we are use to split data into train, test, val')
    parser.add('--test_split', type=str, required=False, help='Test split values')
    parser.add('--val_split', type=str,  required=False, help='Val split values')
    parser.add('--train_split', default=[], type=str, action='append', required=False, help='Train split values (if not set, all excdept test/val values)')

    ######################### Model Training and Prediction #######################################################
    parser.add_argument("--model", type=str, default="mlp")
    parser.add_argument("--model_path", type=str, default="./model.pt")

    # data configs
    parser.add_argument("--dataset", type=str, default="fakedata")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--shuffle", action="store_true")

    # used mostly for fake data, can take it out
    parser.add_argument("--data_dim", type=int, default=24)
    parser.add_argument("--data_len", type=int, default=10000)

    # training configs
    parser.add_argument('--lr', type=float, default=3e-4)

    #prediction
    parser.add("--threshold", type=float, default=0.5)

    # prediction input and output files
    parser.add("--result_output", type=str, default = "../result.csv")

    ######################### Analysis #######################################################
    parser.add_argument("--type", type=str, default="dataset", help='Type of report to generate')

    # data-specific parameters
    parser.add('--slice', default='year', type=str, required=False,
               help='What column to use to slice data for analysis?')
    parser.add('--data_ref', default=[], type=int, action='append', required=False,
               help='List of slices to take as reference data')
    parser.add('--data_eval', default=[], type=int, action='append', required=False,
               help='List of slices to evaluate on')
    parser.add('--numerical_features', default=[], type=str, action='append', required=False,
               help='List of numerical features (for analysis)')
    parser.add('--categorical_features', default=[], type=str, action='append', required=False,
               help='List of categorical features (for analysis)')

    parser.add('--report_path', default='../', type=str, required=False, help='Directory where to store html report?')
    parser.add('--report_full_path', default='', type=str, required=False,
               help="Full path for the report (filename is generated if not provided)")
    parser.add('-html', action='store_true', help='Produce HTML report (otherwise save json report)')

    parser.add('-target_num', action='store_true', required=False,
               help='Is target numerical (as opposed to categorical)')
    parser.add('--prediction_col', default='prediction', type=str, required=False, help='Name of the prediction column')

    # model performance parameters
    parser.add('--reference', type=str, required=False, help='Filename of features/prediction to use as reference')
    parser.add('--test', type=str, required=False,
               help='Filename of features/prediction to use as test (for model drift evaluation)')

    args, unknown = parser.parse_known_args()

    # args.commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('ascii')

    print({k: v for k, v in vars(args).items()})

    if args.input == None and args.password == None:
        args.password = getpass.getpass(prompt='Database password: ', stream=None)
    if len(args.filter_date_from) and len(args.filter_date_to):
        args.filter_date_from = datetime.strptime(args.filter_date_from, '%Y%m%d')
        args.filter_date_to = datetime.strptime(args.filter_date_to, '%Y%m%d')
    
    return args

def write_config(config):
    # save args to args_{date}.json
    t = time.localtime()
    date = time.strftime("%Y-%b-%d_%H-%M-%S", t)
    print(date)

    with open(os.path.join(config.output, f'args_{date}.json'), 'w') as fp:
        fp.write(json.dumps({k: v for k, v in vars(config).items() if k != 'password'}, indent=4))

# Config testing code
if __name__ == '__main__':
    params = read_config()
    write_config(params)
