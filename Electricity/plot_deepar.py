import argparse



import matplotlib

matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42



from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm

import matplotlib.pyplot as pypl

import utils
import attack_utils
from dataloader import *

def sample(params,test_loader):

        # For each test sample
        # Test_loader:
        # test_batch ([batch_size, train_window, 1+cov_dim]): z_{0:T-1} + x_{1:T}, note that z_0 = 0;
        # id_batch ([batch_size]): one integer denoting the time series id;
        # v ([batch_size, 2]): scaling factor for each window;
        # labels ([batch_size, train_window]): z_{1:T}.
        for i, (test_batch, id_batch, v, labels) in enumerate(tqdm(test_loader)):

            if i == 0:
                index = v[:, 0] > 0
                test_batch = test_batch[index]
                id_batch = id_batch[index]
                v = v[index]

                # Prepare batch data
                test_batch = test_batch.permute(1, 0, 2).to(torch.float32).to(params.device)[:,:10]
                id_batch = id_batch.unsqueeze(0).to(params.device)[:,:10]
                v_batch = v.to(torch.float32).to(params.device)[:10]

                batch_size = test_batch.shape[1]
                hidden = model.init_hidden(10)
                cell = model.init_cell(10)

                samples, sample_mu, sample_sigma = model.test(test_batch,
                                                                  v_batch,
                                                                  id_batch,
                                                                  hidden,
                                                                  cell,
                                                                  sampling=True)

                ind = 1
                past_len = 10
                x = range(-past_len,samples.shape[2],1)

                # Plot 10 samples
                for j in range(10):

                    pypl.plot(x[past_len:],samples[j,ind].cpu().detach().numpy())

                np_sample_mu = sample_mu[ind].cpu().detach().numpy()
                np_sample_sigma = sample_sigma[ind].cpu().detach().numpy()
                np_labels = labels[ind,-24-past_len:-24].cpu().detach().numpy()

                pypl.plot(x,np.concatenate([np_labels,np_sample_mu]),color='b')
                pypl.fill_between(x[past_len:],
                                  np_sample_mu - 2*np_sample_sigma,
                                  np_sample_mu + 2*np_sample_sigma,
                                  color='b',
                                  alpha=0.2)
                pypl.axvline(0,color='g',linestyle='dashed')
                pypl.tight_layout()
                pypl.grid()
                pypl.savefig(os.path.join(params.output_folder,"deepar.pdf"))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='elect', help='Name of the dataset')
    parser.add_argument('--data-folder', default='data', help='Parent dir of the dataset')
    parser.add_argument('--model-name', default='base_model', help='Directory containing params.json')
    parser.add_argument('--output_folder', help='Output folder for plots')
    parser.add_argument('--restore-file', default='best',
                        help='Optional, name of the file in --model_dir containing weights to reload before \
                            training')  # 'best' or 'epoch_#'

    args = parser.parse_args()
    model_dir = os.path.join('experiments', args.model_name)
    json_path = os.path.join(model_dir, 'params.json')
    data_dir = os.path.join(args.data_folder, args.dataset)
    assert os.path.isfile(json_path), 'No json configuration file found at {}'.format(json_path)

    params = utils.Params(json_path)
    print(params)
    params.output_folder = os.path.join("attack_logs", args.output_folder)

    if not os.path.exists(params.output_folder):
        os.makedirs(params.output_folder)

    model = attack_utils.set_cuda(params,logger)

    test_set = TestDataset(data_dir, args.dataset, params.num_class)
    test_loader = DataLoader(test_set, batch_size=params.predict_batch,
                             sampler=RandomSampler(test_set),
                             num_workers=4)


    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(model_dir, args.restore_file + '.pth.tar'), model)

    with torch.no_grad():
        sample(params,test_loader)



