from gensim.models import Word2Vec, FastText


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

    product_name = "Samsung"

    candidate_names = [
        "Apple",
        "Samsung",
        "Dell",
        "Casio",
        "Honda",
        "Yamaha",
        "Gucci",
        "Lenovo",
        "Electrolux",
        "Xiaomi",
        "Daikin",
        "Kangaroo",
        "Sony",
        "Downy",
        "TH True Milk",
        "Panasonic",
        "LG",
        "Hitachi",
        "Toshiba",
    ]
    result = []

    for candidate_name in candidate_names:
        sim = model.similarity(product_name, candidate_name)
        result.append((product_name, candidate_name, sim))

    result.sort(key=lambda x: x[2], reverse=True)
    for name1, name2, sim in result:
        print("({} - {}) : {}".format(name1, name2, sim))


if __name__ == "__main__":
    test()
