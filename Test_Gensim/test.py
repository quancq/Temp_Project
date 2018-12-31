from gensim.models import Word2Vec, FastText
import utils
import time
import pandas as pd


def test():
    save_path = "./Model/model_FastText_16000.bin"
    model = FastText.load(save_path)
    print(model)

    # product_name = "Máy tính để bàn Dell Vostro 3668MT- R9 M360 2Gb GDDR5,I7-7700"
    #
    # candidate_names = [
    #     "Máy tính Acer I5 5200U",
    #     "Máy PC để bàn Dell Vos",
    #     "Laptop Asus Intel Core I7 7000",
    #     "Điện thoại Apple 32GB",
    #     "Máy tính Dell Intel Quard core I7-7700",
    #     "Máy tính Lenovo Thinkpad"
    # ]

    # product_name = "Samsung"
    #
    # candidate_names = [
    #     "Apple",
    #     "Samsung",
    #     "Dell",
    #     "Casio",
    #     "Honda",
    #     "Yamaha",
    #     "Gucci",
    #     "Lenovo",
    #     "Electrolux",
    #     "Xiaomi",
    #     "Daikin",
    #     "Kangaroo",
    #     "Sony",
    #     "Downy",
    #     "TH True Milk",
    #     "Panasonic",
    #     "LG",
    #     "Hitachi",
    #     "Toshiba",
    #     "Galaxy Note 11",
    #     "Điện thoại Sam sung Galaxy",
    #     "Máy tính bảng Galaxy Tab",
    # ]

    product_name = "Điện thoại Samsung Galaxy A7 (2018) SM-A750G Black"

    test_dir = "./Dataset/Preprocess/Train"
    test_paths = utils.get_file_paths(test_dir)
    test_df = utils.load_csvs(test_paths)
    candidate_names = test_df["Model"].values.tolist()

    result = []
    start_time = time.time()
    for i, candidate_name in enumerate(candidate_names):
        sim = model.similarity(product_name, candidate_name)
        result.append((product_name, candidate_name, sim))

        if (i+1) % 10 == 0:
            print("Calculate similarity {}/{} done".format(i+1, len(candidate_names)))

    result.sort(key=lambda x: x[2], reverse=True)
    # for name1, name2, sim in result:
        # print("\n{}\n{}\n{}\n".format(name1, name2, sim))
    result_df = pd.DataFrame(result[:1000], columns=["Product_Name", "Candidate_Name", "Similarity"])
    save_path = "./Result/test_similarity.csv"
    utils.save_csv(result_df, save_path)
    exec_time = time.time() - start_time
    print("Time : {:.2f} seconds".format(exec_time))


def test_predict_by_context():
    save_path = "./Model/Word2Vec/model_Word2Vec_16000_20.bin"
    model = Word2Vec.load(save_path)
    print(model)

    context = ["Thông tin", "máy giặt", "lồng ngang", "công nghệ"]

    result = model.predict_output_word(context)
    result.sort(key=lambda x: x[1], reverse=True)
    for word, score in result:
        print("{} - {}".format(word, score))


if __name__ == "__main__":
    # test()
    test_predict_by_context()
