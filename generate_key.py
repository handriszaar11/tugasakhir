import pickle
from pathlib import Path

import streamlit_authenticator as sa

names = ["arya", "benti"]
usernames = ["arya", "benti"]
passwords = []

hashed_passwords = sa.Hasher(passwords).generate()
file_path = Path(__file__).parent / "users.pkl"
with file_path.open("wb") as file:
    pickle.dump(hashed_passwords, file)
    