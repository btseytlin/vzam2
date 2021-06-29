from vzam.models.mobilenetv3 import MobileNetv3


def get_feature_extractor(extractor_name):
    extractors = {
        'MobileNetV3': MobileNetv3,
    }
    return extractors[extractor_name]()