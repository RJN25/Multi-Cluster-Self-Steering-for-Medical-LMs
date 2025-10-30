from calibration.train_ats import train
if __name__ == "__main__":
    # Use MedQA validation (or a held-out slice) for calibration, per ATS being post-hoc.
    train(split="validation")
