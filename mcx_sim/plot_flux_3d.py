import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os
import imageio
import jdata as jd

path = os.path.join("KB_ijv_small_to_large", "LUT",
                    "run_1", "mcx_output", "run_1_0.jnii")
flux = jd.load(path)
flux = flux['NIFTIData']
flux = flux.T
# plt.imshow(flux[:,:,int(flux.shape[2]//2)])
# plt.imshow(flux[:,int(flux.shape[1]//2),:])
# plt.imshow(flux[0,:,:])
temp = flux
ani = []
for t in range(flux.shape[0]):
    flux = temp[t, :, :, :]
    flux = flux[::4, ::4, ::12]
    cmap = ['red', 'salmon', 'sienna', 'silver',
            'tan', 'white', 'violet', 'wheat', 'yellow']

    def explode(data):
        size = np.array(data.shape)*2
        data_e = np.zeros(size - 1, dtype=data.dtype)
        data_e[::2, ::2, ::2] = data
        return data_e
    flux = explode(flux)

    hist, bin_edges = np.histogram(flux)

    colors = np.empty(list(flux.shape) + [4], dtype=np.float32)
    alpha = 0.5
    count = 0
    threshold = 30000
    flat = flux.flatten()
    sort_flux = np.unravel_index(np.argsort(flux, axis=None), flux.shape)
    for i in range(flux.size):

        if flux[sort_flux[0][i], sort_flux[1][i], sort_flux[2][i]] >= threshold:
            cmap = plt.cm.jet(255)
        else:
            cmap = plt.cm.jet(
                1*flux[sort_flux[0][i], sort_flux[1][i], sort_flux[2][i]]/threshold)
            # print(plt.cm.hsv(flux[sort_flux[0][i],sort_flux[1][i],sort_flux[2][i]]/threshold))
        cmap = list(cmap)
        if flux[sort_flux[0][i], sort_flux[1][i], sort_flux[2][i]] >= threshold:
            cmap[-1] = 1
        else:
            cmap[-1] = flux[sort_flux[0][i], sort_flux[1]
                            [i], sort_flux[2][i]]/threshold
        cmap = tuple(cmap)

        colors[sort_flux[0][i], sort_flux[1][i], sort_flux[2][i]] = cmap

    # for i in range(bin_edges.shape[0]-1):
    #     idx = np.where(flux>=bin_edges[i])
    #     for j in range(idx[0].shape[0]):
    #         cmap = plt.cm.hsv(count/hist.sum())
    #         # print(plt.cm.hsv(count/hist.sum()))
    #         cmap = list(cmap)
    #         cmap[-1] = flat[j]/flat.max()
    #         # cmap[-1] = alpha
    #         cmap = tuple(cmap)
    #         colors[idx[0][j],idx[1][j],idx[2][j]] = cmap
    #         count += 1
    #     break
    # colors[flux==1] = [1, 0, 0, alpha]
    # colors[flux==2] = [0, 1, 0, alpha]
    # colors[flux==3] = [0, 0, 1, alpha]
    # colors[flux==4] = [1, 1, 0, alpha]
    # colors[flux==5] = [1, 0, 1, alpha]
    # colors[flux==6] = [0, 1, 1, 0.1]
    # colors[flux==7] = [1, 1, 1, 1]
    # colors[flux==8] = [0, 0, 0, 1]
    # colors[flux==9] = [0.5, 0.5, 0.5, 1]
    edgecolor = [1, 1, 1, 0]

    x, y, z = np.indices(np.array(flux.shape) + 1).astype(float) // 2
    x[0::2, :, :] += 0.05
    y[:, 0::2, :] += 0.05
    z[:, :, 0::2] += 0.05
    x[1::2, :, :] += 0.95
    y[:, 1::2, :] += 0.95
    z[:, :, 1::2] += 0.95

    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(x, y, z, flux, facecolors=colors, edgecolor=edgecolor)
    # surf = ax.plot_surface(X, Y, Z, alpha=0.5, cmap=plt.cm.twilight,
    #                         linewidth=0, antialiased=False)
    ax.view_init(25, 128)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=threshold)
    m = cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
    m.set_array([])
    plt.title(f"{(t+1)*20} ps")
    plt.colorbar(m)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    ani.append(f"gif\\{t}.png")
    plt.savefig(f"gif\\{t}.png")
    plt.show()

with imageio.get_writer('gif\\mygif.mp4', mode="I", fps=2) as writer:
    for filename in ani:
        image = imageio.imread(filename)
        writer.append_data(image)
