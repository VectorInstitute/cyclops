import getpass
import configargparse
import time
import os
import json

def read_config(file = False):
    if not file:
        parser = configargparse.ArgumentParser()
    else:
        parser = configargparse.ArgumentParser(default_config_files = [file])
        
    parser.add('-c', '--config_file', is_config_file=True, help='config file path')

    # database connection parameters
    parser.add("--user", default='koshkinam', type=str, required=False, help='Postgres user')
    # parser.add_argument("--password", default=os.environ['PGPASSWORD'], type=str, required=True, help='Postgres password')
    parser.add("--port", default=5432, type=int, help='Postgres port')
    parser.add("--host", default='db.gemini-hpc.ca', type=str, required=False, help='Postgres host')
    parser.add("--database", default='delirium_v3_0_0', type=str, required=False, help='Postgres database')
    parser.add("--output", type=str, required=True, help='Where should we put the CSV results?')

    # data extraction parameters
    parser.add('-w', action='store_true', help='Write extracted data to disk')
    parser.add('--input', type=str, default=None, required=False, help='Data file to read from instead of database')
    parser.add('--features', default=[], type= str,  action='append', required=False, help='List of features for the model')
    parser.add('--target', default='target', type = str, required = False, help = 'Column we are trying to predict')

    #data analysis parameters
    parser.add('-a', action='store_true', help='Perform analysis?')
    parser.add('--slice', default='year', type=str, required=False, help='What parameter to use to slice data for analysis?')
    parser.add('--data_ref', default=[], type=int,  action='append', required=False, help='List of slices to take as reference data')
    parser.add('--data_eval', default=[], type = int, action='append', required = False, help = 'List of slices to evaluate on (default is all except reference data)')
    parser.add('--numerical_features', default=[], type=str, action='append', required=False, help='List of numerical features (for analysis)')
    parser.add('--categorical_features', default=[], type=str, action='append', required=False, help='List of categorical features (for analysis)')
    parser.add('--report_path', default='../', type=str, required=False, help='Where to store html report?')
    parser.add('--prediction', default='prediction', type=str, required=False, help='Name of the prediction column')

    # model performance parameters
    parser.add('--reference', type=str, required=False, help='Filename of features/prediction to use as reference')
    parser.add('--test', type=str,  required=False, help='Filename of features/prediction to use as test (for model drift evaluation)')


    args, unknown = parser.parse_known_args()

    # args.commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('ascii')

    print({k: v for k, v in vars(args).items()})

    if args.input == None:
        password = getpass.getpass(prompt='Database password: ', stream=None)
        args.password = password

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
