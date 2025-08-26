import h5py, matplotlib.pyplot as plt
with h5py.File("/Users/alex/VSCode/musclex/musclex/tests/mnist-train.hdf5", "r") as f:
    img = f["/image"][5]  # change 0 to any index 0..59999
plt.imshow(img, cmap="gray"); plt.axis("off"); plt.show()