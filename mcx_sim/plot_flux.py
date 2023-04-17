import jdata as jd
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio

"""
MCX output is “W/mm2 = J/(mm2s)”, if it is interpreted as the “energy fluence-rate” [6], 
or “1/(mm2s)”, if the output is interpreted as the “particle fluence-rate”
"""

path = os.path.join("KB_ijv_small_to_large", "LUT",
                    "run_1", "mcx_output", "run_1_0.jnii")
flux = jd.load(path)
flux = flux['NIFTIData']
flux = flux.T


plt.imshow(flux[:, :, int(flux.shape[2]//2)],
           cmap=plt.cm.jet, vmax=0.0001, vmin=0)
plt.colorbar()
plt.show()

plt.imshow(flux[int(flux.shape[0]//2), :, :], cmap=plt.cm.jet)
plt.colorbar()
plt.show()


vol = np.load(os.path.join('ultrasound_image_processing',
              'KB_perturbed_small_to_large.npy'))
vol = vol.T

plt.imshow(vol[:, :, int(vol.shape[2]//2)])
plt.savefig("pic\\sagittal_img.png")
plt.show()
plt.imshow(vol[:, int(vol.shape[1]//2), :])
plt.savefig("pic\\frontal_img.png")
plt.imshow(vol[0, :, :])
plt.savefig("pic\\horizontal_img_0.png")
plt.show()

# sagittal flux gate plot
ani = []
for i in range(flux.shape[0]):
    img1 = plt.imshow(flux[i, :, :, int(flux.shape[2]//2)], cmap=plt.cm.jet)
    plt.colorbar()
    plt.title(f"{i*20} ps")
    plt.axis('off')
    ani.append(f"gif\\{i}.png")
    # plt.show()
    img2 = plt.imshow(vol[:, :, int(vol.shape[2]//2)], alpha=0.5)
    plt.savefig(f"gif\\{i}.png")
    plt.show()

with imageio.get_writer('gif\\mygif.mp4', mode="I", fps=2) as writer:
    for filename in ani:
        image = imageio.imread(filename)
        writer.append_data(image)


# horizontal plane
ani = []
for i in range(flux.shape[0]):
    img1 = plt.imshow(flux[i, :, :], cmap=plt.cm.jet, alpha=0.8)
    plt.colorbar()
    plt.title(f"{i} th depth")
    plt.axis('off')
    ani.append(f"gif\\{i}.png")
    # plt.show()
    # img2 = plt.imshow(vol[i,:,:], alpha=0.5)
    # plt.savefig(f"gif\\{i}.png")
    plt.show()

with imageio.get_writer('gif\\mygif.mp4', mode="I") as writer:
    for filename in ani:
        image = imageio.imread(filename)
        writer.append_data(image)

# frontal plane
plt.imshow(flux[:, int(flux.shape[1]//2), :], cmap=plt.cm.jet, vmax=5000)
plt.axis('off')
plt.colorbar()
plt.savefig("pic\\frontal_flux.png")
plt.show()

plt.imshow(flux[:, int(flux.shape[1]//2), :],
           cmap=plt.cm.jet, vmax=5000, alpha=0.4)
plt.axis('off')
plt.colorbar()
plt.imshow(vol[:, int(vol.shape[1]//2), :], alpha=0.5)
plt.savefig("pic\\frontal_combine.png")
plt.show()


# sagittal plance
plt.imshow(flux[:, :, int(flux.shape[2]//2)],
           cmap=plt.cm.jet, vmax=50000, vmin=0)
plt.axis('off')
plt.colorbar()
plt.savefig("pic\\sagittal_flux.png")
plt.show()

plt.imshow(flux[:, :, int(flux.shape[2]//2)],
           cmap=plt.cm.jet, vmax=30000, vmin=0, alpha=0.8)
plt.axis('off')
plt.colorbar()
plt.imshow(vol[:, :, int(vol.shape[2]//2)], alpha=0.5)
plt.savefig("pic\\sagittal_combine.png")
plt.show()
