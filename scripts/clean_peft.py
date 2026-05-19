import pandas as pd


def clean_df(df, min_chars=20):
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.strip() != ""]
    df = df[df["text"].str.len() >= min_chars]
    return df.reset_index(drop=True)


def main():

    train_path = "../data/ml_train.parquet"
    val_path   = "../data/ml_val.parquet"
    test_path  = "../data/ml_test.parquet"

    train_out  = "../data/ml_train_clean.parquet"
    val_out    = "../data/ml_val_clean.parquet"
    test_out   = "../data/ml_test_clean.parquet"

    seed = 42

    print(" Loading data ".center(50, "*"))

    df_train = pd.read_parquet(train_path)
    df_val   = pd.read_parquet(val_path)
    df_test  = pd.read_parquet(test_path)

    print(" Cleaning ".center(50, "*"))

    df_train = clean_df(df_train)
    df_val   = clean_df(df_val)
    df_test  = clean_df(df_test)

    df_train = df_train.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"Train : {len(df_train):,} rows")
    print(f"Val   : {len(df_val):,} rows")
    print(f"Test  : {len(df_test):,} rows")

    print(" Saving ".center(50, "*"))

    df_train.to_parquet(train_out, index=False)
    df_val.to_parquet(val_out,     index=False)
    df_test.to_parquet(test_out,   index=False)

    print(f"Saved : {train_out}")
    print(f"Saved : {val_out}")
    print(f"Saved : {test_out}")

    print(" Done ".center(50, "*"))


if __name__ == "__main__":
    main()
