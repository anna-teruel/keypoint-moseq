import jax  # type: ignore
import keypoint_moseq as kpms
import numpy as np
import os

project_dir = '/home/vteruel/anna/kpms/kpms_projects/kpms-hierarchy/'
config = lambda: kpms.load_config(project_dir)
keypoint_data_path = '/home/vteruel/anna/kpms/kpms_projects/kpms-hierarchy/dlc/'
coordinates, confidences, bodyparts = kpms.load_keypoints(keypoint_data_path, 'deeplabcut')
data, metadata = kpms.format_data(coordinates, confidences, **config())

pca = kpms.fit_pca(**data, **config())
kpms.save_pca(pca, project_dir)

kpms.print_dims_to_explain_variance(pca, 0.9)
kpms.plot_scree(pca, project_dir=project_dir)
kpms.plot_pcs(pca, project_dir=project_dir, **config())

#kappas = np.logspace(3,7,5)
kappas = [1e06, 1e07, 1e08] 
decrease_kappa_factor = 10
num_ar_iters = 50
num_full_iters = 500

prefix = 'kappa_'

for kappa in kappas:
    print(f"Fitting model with kappa={kappa}")
    model_name = f'{prefix}-{kappa}'
    model = kpms.init_model(data, pca=pca, **config())

    # stage 1: fit the model with AR only
    model = kpms.update_hypparams(model, kappa=kappa)
    from jax_moseq.utils import set_mixed_map_iters
    set_mixed_map_iters(8)
    
    model = kpms.fit_model(
        model,
        data,
        metadata,
        project_dir,
        model_name,
        ar_only=True,
        num_iters=num_ar_iters,
        save_every_n_iters=50
    )[0];

    # stage 2: fit the full model
    #model = kpms.update_hypparams(model, kappa=kappa/decrease_kappa_factor)
    model = kpms.update_hypparams(model, kappa=kappa)
    kpms.fit_model(
        model,
        data,
        metadata,
        project_dir,
        model_name,
        ar_only=False,
        start_iter=num_ar_iters,
        num_iters=num_full_iters,
        save_every_n_iters=25
    );

fig, final_median_durations = kpms.plot_kappa_scan(kappas, project_dir, prefix="my_kappa_scan")
plot_path = os.path.join(project_dir, "kappa_scan_plot.svg")  # Change to .svg
fig.savefig(plot_path, dpi=300, format="svg")  
print(f"Plot saved to: {plot_path}")
