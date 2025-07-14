import os
import imageio


def make_gif_from_train_plots(args, fname: str) -> None:
    treatment = str(int(args.treatment))
    cluster = str(int(args.cluster_num))
    
    png_dir = f"train_plots_{args.dataset}_cluster_{cluster}_treatment_{treatment}/"
    images = []
    sort = sorted(os.listdir(png_dir))
    for file_name in sort[1::1]:
        if file_name.endswith(".png"):
            file_path = os.path.join(png_dir, file_name)
            images.append(imageio.imread(file_path))

    imageio.mimsave("gifs/" + fname, images, duration=0.05)
