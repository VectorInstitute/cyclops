from shift_utils import *
import matplotlib as mpl
import matplotlib.pyplot as plt

datset = sys.argv[1]
dr_technique = sys.argv[3]

# Define results path and create directory.
path = './paper_results/'
path += test_type + '/'
path += datset + '_'
path += sys.argv[2] + '/'
if not os.path.exists(path):
    os.makedirs(path)

samples = [10, 20, 50, 100, 200, 500, 1000]

# Number of random runs to average results over.
random_runs = 5

# Significance level.
sign_level = 0.05

# Whether to calculate accuracy for malignancy quantification.
calc_acc = True

# Define shift types.
if sys.argv[2] == 'small_gn_shift':
    shifts = ['small_gn_shift_0.1',
              'small_gn_shift_0.5',
              'small_gn_shift_1.0']
elif sys.argv[2] == 'medium_gn_shift':
    shifts = ['medium_gn_shift_0.1',
              'medium_gn_shift_0.5',
              'medium_gn_shift_1.0']
elif sys.argv[2] == 'large_gn_shift':
    shifts = ['large_gn_shift_0.1',
              'large_gn_shift_0.5',
              'large_gn_shift_1.0']
elif sys.argv[2] == 'ko_shift':
    shifts = ['ko_shift_0.1',
              'ko_shift_0.5',
              'ko_shift_1.0']
elif sys.argv[2] == '_shift':
    shifts = ['mfa_shift_0.25',
              'mfa_shift_0.5',
              'mfa_shift_0.75']
elif sys.argv[2] == '_shift':
    shifts = ['cp_shift_0.25',
              'cp_shift_0.75']

# -------------------------------------------------
# PLOTTING HELPERS
# -------------------------------------------------
    
linestyles = ['-', '-.', '--', ':']
format = ['-o', '-h', '-p', '-s', '-D', '-<', '->', '-X']
markers = ['o', 'h', 'p', 's', 'D', '<', '>', 'X']
brightness = [1.5, 1.25, 1.0, 0.75, 0.5]
colors = ['#2196f3', '#f44336', '#9c27b0', '#64dd17', '#009688', '#ff9800', '#795548', '#607d8b']


# -------------------------------------------------
# PARAMETERS
# -------------------------------------------------

path = "/path/"
samples = [10, 20, 50, 100, 200, 500, 1000]
# Number of random runs to average results over.
random_runs = 5
# Significance level.
sign_level = 0.05
# Whether to calculate accuracy for malignancy quantification.
calc_acc = True
# Dimensionality Reduction Techniques
dr_techniques = ["PCA","SRP","kPCA","Isomap","BBSDs"]
# Statistical Tests
md_tests = ["MMD","LSDD","LK"]

# -------------------------------------------------
# PIPELINE START
# -------------------------------------------------

mean_shifts_dr_md = np.ones((len(shifts), len(dr_techniques),len(md_tests),len(samples))) * (-1)
std_shifts_dr_md = np.ones((len(shifts), len(dr_techniques),len(md_tests),len(samples))) * (-1)

for si, shift in enumerate(shifts):
    for di, dr_technique in enumerate(dr_techniques):
        for mi, md_test in enumerate(md_tests):
            
            mean_p_vals, std_p_vals = run_shift_experiment(shift, path, dr_technique, md_test, samples, random_runs=5,calc_acc=True)
            mean_shifts_dr_md[si,di,mi,:] = mean_p_vals
            std_shifts_dr_md[si,di,mi,:] = std_p_vals
            
            errorfill(np.array(samples), mean_shifts_dr_md[si,di,mi,:], std_shifts_dr_md[si,di,mi,:], fmt=linestyles[di]+markers[di], color=colorscale(colors[si],brightness[si]), label="%s" % '_'.join([shift,dr_technique,md_test]))
            
plt.xlabel('Number of samples from test')
plt.ylabel('$p$-value')
plt.axhline(y=sign_level, color='k')
plt.legend()
plt.show()
            
            