for file in $LOCAL_DIR/$NAME/*.zip; do
    if [[ $file == *_base.zip ]]; then
        unzip -q -j "$file" -d "$(dirname "$file")"
        echo "Extraction of base file $file is complete."
    elif [[ $file == *bop19.zip ]]; then
        test_all_file="${file/_bop19/_all}"
        if [[ -f $test_all_file ]]; then
            echo "Skipping $file as $test_all_file exists."
        else
            echo "Unzipping $file..."
            unzip -q "$file" -d "$(dirname "$file")"
            echo "Extraction of $file is complete."
        fi
    else
        echo "Unzipping $file..."
        unzip -q "$file" -d "$(dirname "$file")"
        echo "Extraction of $file is complete."
    fi
done