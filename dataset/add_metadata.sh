#!/bin/bash

base_dir="sdv2-lora"
subdirs=$(find "$base_dir" -type d -mindepth 3 -maxdepth 3)

for subdir in $subdirs; do
    if [[ "$subdir" != *"wikiart"* ]]; then
        description="a photo of sks person"
        images=$(find "$subdir" -type f -name "*.jpg" -o -name "*.png" -o -name "*.jpeg")
        if [ -n "$images" ]; then
            metadata_file="$subdir/metadata.csv"
            echo "file_name,text" > "$metadata_file"
            
            for image in $images; do
                image_name=$(basename "$image")
                echo "$image_name,$description" >> "$metadata_file"
            done
        fi
    fi
done