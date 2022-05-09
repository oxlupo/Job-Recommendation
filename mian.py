from codes.train_model import *

if __name__ == '__main__':
    file_path = f"{absolute_path}/dataset/ner/ner_skill.csv"
    train_data, test_data = read_csv_make_dataset(file_path)
    model = train_model(args=args,
                        dataset=pd.read_csv(file_path),
                        train_data=train_data,
                        test_data=test_data)

    evaluate_model(model=model,
                   test_data=test_data)

    predication_model(model=model,
                      sentences="im a deep learning engineer in apple company")