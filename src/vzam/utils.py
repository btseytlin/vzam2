from vzam.models.mobilenetv3 import MobileNetv3

def l_normalize(x, p=2):
    norm = x.norm(p=p, dim=1, keepdim=True)
    x_normalized = x.div(norm.expand_as(x))
    return x_normalized

def get_feature_extractor(extractor_name):
    extractors = {
        'MobileNetV3': MobileNetv3,
    }
    return extractors[extractor_name]()

def fname(txt):
    return '.'.join(txt.split('.')[:-1])
