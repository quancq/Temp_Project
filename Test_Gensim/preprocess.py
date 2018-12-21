import utils
import string
from pyvi import ViTokenizer
import sys
import unicodedata
import re
import html


MAX_WORD_LENGTH = 20


def pre_process(input):

    # Decode HTML entity
    output = html.unescape(input)

    # Remove HTML tags
    output = re.sub("<.*?>", "", output)

    # Remove punctuation
    retain_chars = ['.']
    remove_punct_map = dict.fromkeys(i for i in range(sys.maxunicode)
                                     if unicodedata.category(chr(i)).startswith('P') and
                                     chr(i) not in retain_chars)
    output = output.translate(remove_punct_map)

    # Remove duplicate white spaces
    output = re.sub(r"\s+", " ", output)

    # Remove duplicate periods
    output = re.sub(r"\.(\s*\.)+", ".", output)

    # Reomve words too long
    output = " ".join([w for w in output.split() if len(w) <= MAX_WORD_LENGTH])

    # Remove html tags

    return output


def tokenize(input):
    output = ViTokenizer.tokenize(input)

    return output


def remove_stopwords(input, stopwords=[]):
    # stopwords_path = "./Dataset/vi_stopwords.txt"
    # stopwords = utils.load_list(stopwords_path)
    # print("\nStopwords : ", stopwords)

    output = " ".join([w for w in input.split() if w not in stopwords])

    return output


def pre_process_pipeline(input, stopwords):
    output = pre_process(input)
    tokens = tokenize(output)
    output = remove_stopwords(tokens, stopwords)

    return output


def test_preprocess_pipeline(input=None):
    # input = " *& . xinchaodaylaturatdaivocungblah. ai đó ở lại nhé, thì tốt bao nhiêu thì với là ở đây để ờ .- Kiểu dáng: Hút   mùi     ống khói. - Tổng công suất: 200W (công suất motor: 196W; công suất đèn LED: 2W x 2). - Điện thế: 220V-50Hz. - Màu sắc: Inox. - An toàn, thuận tiện trong việc sử dụng.. - Kích thước: 930 x 500 x 7000mm. . . . . Hình ảnh tham khảo: Hút mùi kính cong Sunhouse SHB6629. . . . . . . . . . . . . . . . . . Thông số kỹ thuật: Hút mùi kính cong Sunhouse SHB6629. . . . . . . . . . Thông tin: Hút mùi kính cong Sunhouse SHB6629. . . Mô tả về máy hút khử mùi Sunhouse SHB6629. - Hút và khử mùi kính cong Sunhouse SHB6629 thiết kế với phần vỏ bọc thân máy bằng inox phủ một lớp màu trắng ghi tinh tế, tạo nên một không gian sang trọng, thoáng mát cho một căn bếp hiện đại cho một gia đình hiện đại.. - Máy hoạt động khá tốt với một bộ phận motor siêu bền cùng với 3 chế độ hút khác nhau: thấp, trung bình và cao khá tiện cho người sử dụng có thể điều chỉnh mức độ hút sao cho phù hợp với lượng khói, mùi tạo ra từ việc nấu ăn.. -Máy hút mùi có 4 phím điều khiển giúp người sử dụng có thể điều chỉnh hoạt động hút của máy một cách phù hợp để cho máy có thể hoạt động một cách hiệu quả nhất.. - Cùng với đó, máy hút khử mùi Sunhouse SHB6629 thiết kế với chức năng hút và khử mùi tiện dụng. Khói từ bếp, thức ăn tạo ra sẽ được hút toàn bộ lên trên máy và được khử sạch trước khi đưa ra ngoài.. - Máy hút hoạt động với công suất 180W trong đó phần motor của máy hoạt động với công suất 170W, 2 bóng đèn hiển thị hoạt động với công suất 5W cho một bóng đèn giúp tiết kiệm khá nhiều chi phí điện năng tiêu thụ cho người sử dụng.. - Ngoài ra, máy làm sạch mùi nhà bếp Sunhouse SHB6629 có công suất hút 950 m³/h có khả năng hút khói, làm sạch mùi thức ăn từ bếp khiến gia đình bạn cảm thấy khó chịu một cách nhanh chóng, nhưng chỉ tạo ra một lượng độ ồn nhỏ hơn 57 dB không làm ảnh hưởng tới sinh hoạt của bạn và cả gia đình.. . . Thông số kỹ thuật của máy hút mùi Sunhouse SHB6629. - Kích thước: (700/900mm x 500mm x 550mm). - Màu sắc: Trắng inox. - Điện thế: 220v~50Hz. - Công suất hút: 950m³/h. - Kiểu dáng: Hút mùi kính cong. . . . . . Tư Vấn Mua Máy hút khói - hút mùi. . . . . . "
    if input is None:
        input = " . - Loại máy: Lồng đứng. - Dung lượng giặt: 10 Kg. - Điện năng tiêu thụ: 480 W. - Lượng nước tiêu thụ: 165 Lít. - Tốc độ vắt: 700 vòng / Phút. - Hiệu ứng thác nước đôi, Giặt cô đặc bằng bọt khí. - Mâm giặt Hybrid Powerful, kháng khuẩn. - Tính năng Fragrance Course. - Làm khô lồng giặt, Lưới chống chuột. . . . . Hình ảnh tham khảo: Máy giặt 10 Kg Toshiba B1100GV(WD) lồng đứng, nắp màu xám đồng. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . Thông số kỹ thuật: Máy giặt 10 Kg Toshiba B1100GV(WD) lồng đứng, nắp màu xám đồng. . . . ĐẶC ĐIỂM SẢN PHẨM. . . Kiểu máy giặt. . Cửa trên. . . . Kiểu lồng giặt. . Lồng đứng. . . . Khối lượng giặt. . 10Kg. . . . Tốc độ quay vắt (vòng/ phút). . 700 vòng/phút. . . . Truyền động. . Bằng dây Curoa. . . . Lượng nước tiêu thụ (lít). . 165 lít. . . . Công suất (W). . 480. . . . Bảng điều khiển. . Anh - Việt. . . CÔNG NGHỆ. . . Chế độ giặt. . 6 chương trình giặt. . . . Công nghệ giặt. . Mâm giặt Hybrid Powerful. . . . Thiết kế lồng giặt. . Lồng giặt ngôi sao pha lê. . . . Tính năng. . Vắt khô nhanh, Tiết kiệm nước, Tiết kiệm điện, Hẹn giờ. . . THÔNG TIN CHUNG. . . Số người sử dụng. . trên 6 người (trên 8.5 Kg). . . . Chất liệu lồng giặt. . Thép không gỉ. . . . Trọng lượng (kg). . 39 kg. . . . Xuất xứ. . Thái Lan. . . . Bảo hành. . 24 tháng. . . . . . . . . Thông tin: Máy giặt 10 Kg Toshiba B1100GV(WD) lồng đứng, nắp màu xám đồng. . . Máy giặt sang trọng, thiết kế đẹp mắt. Được thiết kế mới mẻ cùng màu sắc lạ mắt, máy giặt Toshiba AW-B1100GV hứa hẹn sẽ đem lại sự tươi mới cho gia đình bạn. Cùng khối lượng giặt được là 10 kg, máy giặt Toshiba sẽ phù hợp hơn cho các gia đình có từ 6 thành viên trở lên.. . . . Mâm giặt Hybrid Powerful kháng khuẩn cao. Máy giặt Toshiba AW-B1100GV có mâm giặt Hybrid Powerful với chức năng kháng khuẩn cao nhờ việc giữ vệ sinh đáy mâm giặt, đồng thời tạo luồng nước xoáy đánh được các vết bẩn bám lâu ngày.. . . Công nghệ Mega Power Wash mạnh mẽ. Công nghệ đặc biệt này sẽ đem lại cho máy giặt Toshiba AW-B1100GV khả năng giặt tốt hơn:. + Hiệu ứng thác nước đôi:. Được tạo nên nhờ cánh quạt quay dưới mâm, hai dòng nước liên tục tuần hoàn của máy giặt Toshiba AW-B1100GV sẽ tạo điều kiện cho hiệu quả giặt và xả được tốt hơn.. + Lồng giặt ngôi sao pha lê:. Có các gờ nổi trên thành lồng giặt, giúp cọ xát để làm sạch áo quần.. + Mâm giặt Mega Power:. Có khả năng tạo dòng nước ba chiều cực mạnh, đánh bại những vết bẩn cứng đầu.. . . Khả năng giặt cô đặc bằng bọt khí. Với công nghệ giặt cô đặc bằng bọt khí, bột giặt sẽ được đánh tan thành những hạt bọt nhỏ hơn, hạn chế tình trạng cặn bột giặt bám trên áo quần, cũng như làm sạch sâu sợi vải hơn.. . . Khả năng tiết kiệm nước và điện năng cao. Máy giặt Toshiba AW-B1100GV có thể tránh hao phí điện năng và nước tốt, để người dùng có được khoản tiết kiệm nho nhỏ cho gia đình mình.. . . Đa dạng chương trình giặt. Nhiều chương trình giặt khác nhau đem đến cho máy giặt Toshiba AW-B1100GV khả năng giặt phù hợp với nhiều loại vải hơn. Từ đó, bạn sẽ yên tâm chọn được cho mình từng chương trình phù hợp cho các lần giặt khác nhau.. . . Hệ thống khe hút khí vòng cung vắt cực khô. Với hệ thống khe hút khí vòng cung, máy giặt Toshiba AW-B1100GV có thể vắt cực khô trong thời gian nhanh chóng, tiết kiệm thời gian cho gia đình bạn.. . . . . . . Tư Vấn Mua Máy giặt. . . . . . "

    print("\n\n======= Input ========\n\n")
    print(input)

    output = pre_process(input)
    print("\n\n======== Preprocess ========\n\n")
    print(output)

    tokens = tokenize(output)
    print("\n\n========Tokens========\n\n")
    print(tokens)

    stopwords = utils.load_list("./Dataset/vi_stopwords.txt")
    output = remove_stopwords(tokens, stopwords)
    print("\n\n========Remove stopwords========\n\n")
    print(output)


def main():
    test_preprocess_pipeline()


if __name__ == "__main__":
    pass
    main()
