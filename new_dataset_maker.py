import os


def generate_txt_file(root_dir, output_filename):
    with open(output_filename, 'w') as outfile:
        # Iterate over all class directories in the root directory
        for class_dir in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_dir)

            # Ensure it's a directory
            if os.path.isdir(class_path):
                # Iterate over all images in the class directory
                for img_file in os.listdir(class_path):
                    # Write the class and image name to the output file
                    outfile.write(f"{class_dir}/{img_file}\n")


# Paths to the train and val directories
train_dir = "/data/dataset/imagenet_small/train"
val_dir = "/data/dataset/imagenet_small/val"

# Generate the txt files
generate_txt_file(train_dir, "small_train.txt")
generate_txt_file(val_dir, "small_val.txt")