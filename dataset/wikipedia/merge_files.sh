WIKI_DIR=($1)
OUT_DIR=($2)
OUT_FILENAMES=($3)

mkdir -p $OUT_DIR

find "$WIKI_DIR" -type f -print0 |
  while IFS= read -r -d '' line; do
    filename=$(echo "$line" | rev | cut -d'/' -f 1 | rev)
    subfilename=$(echo "$line" | rev | cut -d'/' -f 2 | rev)
    prefix="${subfilename}_${filename}"
    new_name=$(echo "$line")
    echo "Procesing $prefix, $filename, $new_name"
    cat $new_name >>$OUT_DIR/$OUT_FILENAMES.json
  done
