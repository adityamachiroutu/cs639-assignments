# Step 0. Change this to your campus ID
CAMPUSID='9086096394'
mkdir -p $CAMPUSID

# Step 1. (Optional) Any preprocessing step, e.g., downloading pre-trained word embeddings
GLOVE_ZIP="glove.6B.zip"
GLOVE_URL="https://nlp.stanford.edu/data/glove.6B.zip"

if ! [ -f "$GLOVE_ZIP" ] || ! unzip -t "$GLOVE_ZIP" >/dev/null 2>&1; then
    rm -f "$GLOVE_ZIP"
    if command -v wget >/dev/null 2>&1; then
        wget -O "$GLOVE_ZIP" "$GLOVE_URL"
    else
        curl -L -o "$GLOVE_ZIP" "$GLOVE_URL"
    fi
fi
unzip -o "$GLOVE_ZIP"

# Use venv python if available
PYTHON_BIN="python"
if [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
fi

# Step 2. Train models on two datasets.
##  2.1. Run experiments on SST
PREF='sst'
"$PYTHON_BIN" main.py \
    --train "data/${PREF}-train.txt" \
    --dev "data/${PREF}-dev.txt" \
    --test "data/${PREF}-test.txt" \
    --emb_file "glove.6B.300d.txt" \
    --emb_size 300 \
    --max_train_epoch 6 \
    --dev_output "${CAMPUSID}/${PREF}-dev-output.txt" \
    --test_output "${CAMPUSID}/${PREF}-test-output.txt" \
    --model "${CAMPUSID}/${PREF}-model.pt"

##  2.2 Run experiments on CF-IMDB
PREF='cfimdb'
"$PYTHON_BIN" main.py \
    --train "data/${PREF}-train.txt" \
    --dev "data/${PREF}-dev.txt" \
    --test "data/${PREF}-test.txt" \
    --emb_file "glove.6B.300d.txt" \
    --emb_size 300 \
    --dev_output "${CAMPUSID}/${PREF}-dev-output.txt" \
    --test_output "${CAMPUSID}/${PREF}-test-output.txt" \
    --model "${CAMPUSID}/${PREF}-model.pt"


# Step 3. Prepare submission:
##  3.1. Copy your code to the $CAMPUSID folder
for file in 'main.py' 'model.py' 'vocab.py' 'setup.py' 'run_exp.sh'; do
	cp $file ${CAMPUSID}/
done
##  3.2. Compress the $CAMPUSID folder to $CAMPUSID.zip (containing only .py/.txt/.pdf/.sh files)
"$PYTHON_BIN" prepare_submit.py ${CAMPUSID} ${CAMPUSID}
##  3.3. Submit the zip file to Canvas (https://canvas.wisc.edu/courses/292771/assignments)! Congrats!
