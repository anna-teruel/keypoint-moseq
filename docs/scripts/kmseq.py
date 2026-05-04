
import keypoint_moseq as kpms

project_dir = '/home/vteruel/anna/kpms/kpms_projects/kpms-3cham-wanhui/'
config = lambda: kpms.load_config(project_dir)
keypoint_data_path = '/home/vteruel/anna/kpms/kpms_projects/kpms-3cham-wanhui/dlc/'

###### Load data
coordinates, confidences, bodyparts = kpms.load_keypoints(keypoint_data_path, 'deeplabcut') 
data, metadata = kpms.format_data(coordinates, confidences, **config()) 

###### Fit PCA
pca = kpms.fit_pca(**data, **config())
kpms.save_pca(pca, project_dir)

kpms.print_dims_to_explain_variance(pca, 0.9)
kpms.plot_scree(pca, project_dir=project_dir)
kpms.plot_pcs(pca, project_dir=project_dir, **config())

##### Define kappa
kappa = 1e4
decrease_kappa_factor = 10
num_ar_iters = 50
num_full_iters = 750

###### Fit AR-HMM

print(f"Fitting model with initial kappa={kappa}")
model = kpms.init_model(data, pca=pca, **config())
model = kpms.update_hypparams(model, kappa=kappa)

model, model_name = kpms.fit_model(
    model, data, metadata, project_dir,
    ar_only=True, num_iters=num_ar_iters)

###### Fitting the full model

# load model checkpoint
model, data, metadata, current_iter = kpms.load_checkpoint(
    project_dir, model_name, iteration=num_ar_iters)

# modify kappa to maintain the desired syllable time-scale
kappa_new = kappa / decrease_kappa_factor
print(f"Updating kappa for full model: {kappa} → {kappa_new}")
model = kpms.update_hypparams(model, kappa=kappa_new)

# run fitting for an additional 500 iters

from jax_moseq.utils import set_mixed_map_iters
set_mixed_map_iters(12)

model = kpms.fit_model(
    model, data, metadata, project_dir, model_name, ar_only=False, parallel_message_passing=True,  
    start_iter=current_iter, num_iters=current_iter+num_full_iters, jitter=1e-3)[0]

###### Sort syllables by frequency

kpms.reindex_syllables_in_checkpoint(project_dir, model_name);

###### Extract model results

model, data, metadata, current_iter = kpms.load_checkpoint(project_dir, model_name)

# extract results
results = kpms.extract_results(model, metadata, project_dir, model_name)

# optionally save results as csv
kpms.save_results_as_csv(results, project_dir, model_name)

