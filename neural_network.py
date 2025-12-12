# %% - Neural Network
## Data setup (split into x, y, and validation)
# drop any na lines
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna().reset_index(drop=True)

label_col = "outcome"

# Categorical features (will be one-hot)
cat_cols = ["p1_dg_id", "p2_dg_id", "p3_dg_id", "course_num_p1"]

# We want our ID columns to not be numeric valued
id_cols = ["p1_dg_id", "p2_dg_id", "p3_dg_id"]

# Data Train/Val/Test splits 
train_df_full, test_df = train_test_split(
    df, test_size=0.15, random_state=42, stratify=df[label_col]
)
train_df, val_df = train_test_split(
    train_df_full, test_size=0.20, random_state=42, stratify=train_df_full[label_col]
)

# Possible outcomes. Subtract 1 so we have outcomes [0,1,2] which is more friendly then [1,2,3] player numbers
y_train = train_df[label_col].values.astype("int32") -1
y_val   = val_df[label_col].values.astype("int32") -1
y_test  = test_df[label_col].values.astype("int32") -1

# One-Hot Encoding for Player IDs and Course ID categorical variables (based on the player)
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

X_train_cat = ohe.fit_transform(train_df[cat_cols])
X_val_cat   = ohe.transform(val_df[cat_cols])
X_test_cat  = ohe.transform(test_df[cat_cols])

# Numeric Features: everything except IDs, cat_cols, and label
feature_cols = [
    c for c in df.columns 
    if c not in id_cols + cat_cols + [label_col]
]

X_train_num = train_df[feature_cols].values
X_val_num   = val_df[feature_cols].values
X_test_num  = test_df[feature_cols].values

# Scale numeric features
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train_num)
X_val_num   = scaler.transform(X_val_num)
X_test_num  = scaler.transform(X_test_num)

# Put all the categorical and numeric feautures together
X_train = np.concatenate([X_train_num, X_train_cat], axis=1)
X_val   = np.concatenate([X_val_num,   X_val_cat],   axis=1)
X_test  = np.concatenate([X_test_num,  X_test_cat],  axis=1)

## Neural Network Model
num_classes = 3
input_dim = X_train.shape[1]

model = Sequential([
    Input(shape=(input_dim,)),
    Dense(128, activation="relu"),
    Dropout(0.2),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(3, activation="softmax")
])

model.compile(
    optimizer=optimizers.SGD(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callback = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=100,
    validation_data=(X_val, y_val),
    callbacks=[callback]
)

test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)

plt.plot(history.history['accuracy'], label="Accuracy")
plt.plot(history.history['val_accuracy'], label="Val_accuracy")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()
