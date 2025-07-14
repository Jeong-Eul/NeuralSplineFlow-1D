import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pandas as pd 
import argparse
import pickle

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

from neural_spline_flow1D import NeuralSplineFlow1D
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from torch.distributions import Normal
from IPython.display import display, Markdown
from util_nf import *
from gif import *


torch.manual_seed(9492)
torch.cuda.manual_seed(9492)
torch.cuda.manual_seed_all(9492)
np.random.seed(9492)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cancer', choices=['Cancer', 'MIMIC']) #
    parser.add_argument('--treatment', type=int, default=0, help='treatment option')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=300, help='epoch')
    parser.add_argument('--training', type=lambda x: x.lower() == 'true', default=True, help='if True, normalizing flow is trained, else, inference mode') #
    parser.add_argument('--cluster', type=int, default=2, help='cluster')
    parser.add_argument('--features', type=int, default=1, help='num of features you interest in')
    parser.add_argument('--num_bins', type=int, default=10, help='num of bins you interest in')
    parser.add_argument('--tau', type=int, default=10, help='rist volume')
    parser.add_argument('--risk_threshold', type=float, default=0.3, help='risk score')


    args, unknown = parser.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == 'Cancer':
        
        if args.training:
            # potential_outcome_df = pd.read_csv(f'processed_potential_outcome_{args.dataset}.csv')
            potential_outcome_df = pd.read_csv('../data/processed_potential_outcome.csv')
            valid_df = potential_outcome_df[potential_outcome_df['prev_output']>5]
            
            del potential_outcome_df
            
            # KMeans
            prev_outputs = valid_df[['prev_output']].copy()

            n_clusters = 3
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
            prev_outputs['cluster'] = kmeans.fit_predict(prev_outputs)

            cluster_counts = prev_outputs['cluster'].value_counts().sort_index()
            cluster_centers = kmeans.cluster_centers_

            cluster_summary = pd.DataFrame({
                'Cluster ID': cluster_counts.index,
                'Sample Count': cluster_counts.values,
                'Center (prev_output)': cluster_centers.flatten()
            })
            
            display(cluster_summary)
            print('Select cluster number:')
            cluster_num = input()
            args.cluster_num = int(cluster_num)
            
            centroid = cluster_summary[cluster_summary['Cluster ID'] == args.cluster_num]['Center (prev_output)'].values
            
            cluster_data = prev_outputs[prev_outputs['cluster'] == args.cluster_num]
            filtered_df = valid_df.loc[cluster_data.index]
            
            df = filtered_df.copy()
            ssc = MinMaxScaler()
            df[f'y{args.treatment}_scaled']=ssc.fit_transform(df[f'y{args.treatment}'].values.reshape(-1, 1))

            # hyperparameter setting
            args.batch_size = len(df)
            
            print('--Dataset: %s, Cluster: %d, Potential outcome: %d--' % (args.dataset, args.cluster_num, int(args.treatment)))

            y_min =  df[f'y{args.treatment}_scaled'].values.min()
            y_max = df[f'y{args.treatment}_scaled'].values.max()

            dataset = TensorDataset(torch.tensor(df[f'y{args.treatment}_scaled'].values).unsqueeze(-1))
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

            model = NeuralSplineFlow1D(args.features, args.num_bins).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
            base_dist = torch.distributions.Normal(0, 1)
            
            model.train()
            best_loss = 100
            patience = 0
            patience_limit = 500
                
            for epoch in range(1, args.epochs + 1):
                
                epoch_loss = 0.0
                
                for batch in dataloader:
                    x_batch = batch[0].to(device).float()   # shape: (batch_size, 1) or (batch_size,)
                    
                    optimizer.zero_grad()
                    
                    z, logdet = model._elementwise_inverse(x_batch) # x -> z_0
                    logprob = base_dist.log_prob(z).sum(dim=-1)
                    loss = -(logprob + logdet).mean()
                    

                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    
                N = len(dataloader.dataset)
                avg_loss = epoch_loss / N
                    
                if epoch % 10 == 0:
                    print(f"Epoch {epoch:03d}: "
                        f"Loss={avg_loss:.4f} ")

                if epoch == 1 or epoch % 1 == 0:
                    plot_training_1d_kde(args, ssc, device, base_dist, df[f'y{args.treatment}'].values, model, epoch, args.treatment)
                    
                
                current_loss = avg_loss  # 혹은 avg_val_loss

                if current_loss < best_loss:
                    best_loss = current_loss
                    patience = 0
                    
                    treatment = str(int(args.treatment))
                    cluster = str(int(args.cluster_num))
                    
                    model_path = f'{args.dataset}_PO_distribution/'
                    
                    save_dict = {
                        'model_state_dict': model.state_dict(),
                        'scaler': ssc,
                        'bins': args.num_bins,
                        'centroid':centroid,
                        'features': args.features
                    }
                    
                    os.makedirs(model_path, exist_ok = True)
                    torch.save(save_dict, model_path+ f'po_treatment_{treatment}_cluster_{cluster}' +'.pt')
                    
                else:
                    patience += 1
                    # print(f"[Epoch {epoch}] No improvement. Patience: {patience}/{patience_limit}")

                if patience > patience_limit:
                    print(f"Stopping early at epoch {epoch}. Best loss: {best_loss:.4f}")
                    break
                    
                    
                    
            # Generate and display an animation of the training progress.
            make_gif_from_train_plots(args, f'{args.dataset}_po_treatment_{treatment}_cluster_{cluster}.gif')
            sum_probability(model, base_dist, device)
            
            
            
        else:
            print('Decision making process........')
            model_path = f'{args.dataset}_PO_distribution/'
            
            # potential_outcome_df = pd.read_csv(f'processed_potential_outcome_{args.dataset}.csv')
            potential_outcome_df = pd.read_csv('../data/processed_potential_outcome.csv')
            valid_df = potential_outcome_df[potential_outcome_df['prev_output']>5]
            
            del potential_outcome_df
            
            # KMeans
            prev_outputs = valid_df[['prev_output']].copy()

            n_clusters = 3
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
            prev_outputs['cluster'] = kmeans.fit_predict(prev_outputs)

            cluster_counts = prev_outputs['cluster'].value_counts().sort_index()
            cluster_centers = kmeans.cluster_centers_

            cluster_summary = pd.DataFrame({
                'Cluster ID': cluster_counts.index,
                'Sample Count': cluster_counts.values,
                'Center (prev_output)': cluster_centers.flatten()
            })
            
            display(cluster_summary)
            
            print('Please enter the cluster number you want to analyze:')
            
            cluster = input()
            
            print('Please enter the treatment option you want to analyze:')
            
            print('Chemo only: %d, Radio only: %d, Chemo & Radio: %d--' % (2, 3, 4))

            treatment = input()
                        
            treatment = str(int(treatment))
            cluster = str(int(cluster))
            
            if treatment == 2:
                prob_variable = 'Chemo'
            elif treatment == 3:
                prob_variable = 'Radio'
            else:
                prob_variable = 'Chemo & Radio'
            
            print('--Dataset: %s, Cluster: %s, Potential outcome: %s--' % (args.dataset, cluster, treatment))
            print('--Starting decision making process...--')
            
            base_dist = torch.distributions.Normal(0, 1)
            
            # treated
            
            checkpoint = torch.load(model_path+ f'po_treatment_{treatment}_cluster_{cluster}' +'.pt')
            
            num_bins = checkpoint['bins']
            features = checkpoint['features']
            ssc_t = checkpoint['scaler']
            centroid = checkpoint['centroid']
            
            
            treated_model = NeuralSplineFlow1D(features, num_bins).to(device)
            treated_model.load_state_dict(checkpoint['model_state_dict'])
            
            z_treated = base_dist.sample((5000,)).to(device) # sampling
            y_treated, _ = treated_model._elementwise_forward(z_treated.unsqueeze(-1))
            y_np_treated = y_treated.squeeze().cpu().detach().numpy()
            
            y_hat_treated = ssc_t.inverse_transform(y_np_treated.reshape(-1, 1))
            
            # not treated
            
            checkpoint = torch.load(model_path+ f'po_treatment_1_cluster_{cluster}' +'.pt')

            num_bins = checkpoint['bins']
            features = checkpoint['features']
            ssc_nt = checkpoint['scaler']
            
            n_treated_model = NeuralSplineFlow1D(features, num_bins).to(device)
            n_treated_model.load_state_dict(checkpoint['model_state_dict'])
            
            z_n_treated = base_dist.sample((5000,)).to(device) # sampling
            y_n_treated, _ = n_treated_model._elementwise_forward(z_n_treated.unsqueeze(-1))
            y_np_n_treated = y_n_treated.squeeze().cpu().detach().numpy()
            
            y_hat_n_treated = ssc_nt.inverse_transform(y_np_n_treated.reshape(-1, 1))
            
            
            plt.figure(figsize=(6, 4))
            sns.kdeplot(y_hat_treated.squeeze(), label=f'P(Y[{prob_variable}] = y)', color='blue')     # no treatment
            sns.kdeplot(y_hat_n_treated.squeeze(), label='P(Y[Not Treated] = y)', color='orange')   # treated
            plt.title("Marginal Potential Outcome Distributions")
            plt.xlabel("y")
            plt.ylabel("Density")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            
            png_dir = model_path + f'po_treatment_{treatment}_cluster_{cluster}_marginal_distribution'
        
            os.makedirs(png_dir, exist_ok=True)
            plt.savefig(png_dir + "/marginal_distribution.png")
            plt.close()
            
            
            # Decision making
            
            mean_treated, var_treated = compute_summary_stats(y_hat_treated)
            mean_n_treated, var_n_treated = compute_summary_stats(y_hat_n_treated)
            
            deviation_treated = np.sqrt(var_treated)
            deviation_n_treated = np.sqrt(var_n_treated)
            
            # P(y < tau)

            p_treated_lt_tau = np.mean(y_hat_treated < args.tau)
            p_n_treated_lt_tau = np.mean(y_hat_n_treated < args.tau)
            
            # P(y(1) > y(0)) adverse effect
            p_harm = np.mean(y_hat_treated > y_hat_n_treated)
            
            # Conditional average treatment effect
            
            cate = (mean_treated - mean_n_treated)
            
            # 95% CI of treatment effect Y(1) - Y(0)
            treatment_effects = y_hat_treated - y_hat_n_treated
            ci_lower, ci_upper = np.percentile(treatment_effects, [2.5, 97.5])
            
            # Generate report
            report_lines = [
                "=== Treatment Decision Support Report ===\n",
                
                f"Target Patient Group: Cluster {cluster} - centroid of cancer volumes wihtin cluster = {centroid}\n"
                
                "1. Summary Statistics",
                f"- Treated Group: Mean = {mean_treated:.2f} cm^3, Standard Deviation = {deviation_treated:.2f}",
                f"- Control Group: Mean = {mean_n_treated:.2f} cm^3, Standard Deviation = {deviation_n_treated:.2f}\n",
                
                "2. Probability of Tumor Shrinking Below Threshold",
                f"- Threshold (tau): {args.tau} cm^3",
                f"- P(Y[{prob_variable}] < {args.tau}cm^3) = {p_treated_lt_tau:.3f}",
                f"- P(Y[Not treated] < {args.tau}cm^3) = {p_n_treated_lt_tau:.3f}\n",
                
                "3. Risk of Harmful Treatment Effect",
                f"- P(Y(1) > Y(0)) = {p_harm:.3f}",
            ]
            
            percent_rt = int(args.risk_threshold * 100)
            if p_harm > args.risk_threshold:
                    report_lines.append(f"=> Treatment({prob_variable}) is NOT recommended due to high risk of harm (>{percent_rt}%).\n")
            else:
                report_lines.append(f"=> Treatment({prob_variable}) is considered safe under current threshold of {percent_rt}%.\n")

            
            report_lines += [
                f"4. Estimated Conditional Average Treatment Effect ((Y(1) - Y(0)|Cluster = {cluster}))",
                f": {cate:.3f}\n",
            ]
            
            
            report_lines += [
                f"5. Estimated CATE (Y(1) - Y(0))- 95% Confidence Interval: [{ci_lower:.2f}, {ci_upper:.2f}]\n",
                f"Note: Lower values are better (smaller tumor size).\n",
                f"Please refer to the marginal_distribution.png for the visual comparison of distributions."
            ]
            
            
            # Save report
            report_path = os.path.join(png_dir, "treatment_decision_report.txt")
            with open(report_path, "w") as f:
                for line in report_lines:
                    f.write(line + "\n")
                    
            print('Successfully saved.')