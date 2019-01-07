def get_captcha(pre_list, characters):
    """
    :param: pre_list type numpy, like[[1,2,3,4]]
    """
    captcha = ''
    for character in pre_list[0]:
            captcha += characters[character]
    return captcha