plt.imshow(flux[:, :, int(flux.shape[2]//2)])
plt.imshow(flux[:, int(flux.shape[1]//2), :])
plt.imshow(flux[0, :, :])