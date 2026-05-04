
import keypoint_moseq as kpms
project_dir = '/home/vteruel/anna/kpms/kpms_projects/kpms-3cham'
config = lambda: kpms.load_config(project_dir)
model_name = '2025_07_10-18_30_37'
keypoint_data_path = '/home/vteruel/anna/kpms/kpms_projects/kpms-3cham/dlc/'
coordinates, confidences, bodyparts = kpms.load_keypoints(keypoint_data_path, 'deeplabcut') #load

# load the most recent model checkpoint and pca object
model = kpms.load_checkpoint(project_dir, model_name)[0]

# load new data (e.g. from deeplabcut)
new_data = '/home/vteruel/anna/kpms/kpms_projects/kpms-3cham/dlc' 
coordinates, confidences, bodyparts = kpms.load_keypoints(new_data, 'deeplabcut')
data, metadata = kpms.format_data(coordinates, confidences, **config())

from jax_moseq.utils import set_mixed_map_iters
set_mixed_map_iters(16)

# apply saved model to new data
results = kpms.apply_model(model, data, metadata, project_dir, '2025_07_10-18_30_37', **config())

# optionally rerun `save_results_as_csv` to export the new results
kpms.save_results_as_csv(results, project_dir, model_name)
