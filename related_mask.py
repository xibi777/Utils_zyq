for i in range(len(targets)):
    for j in range(targets[i]["masks"].shape[0]):
        mask_bool = targets[i]["masks"][j]
        mask = mask_bool.to(filters).unsqueeze(0).unsqueeze(0)
        mask_ = F.conv2d(mask, filters, stride=4)
        mask_ = mask_.squeeze(0).squeeze(0).to(mask_bool)
        feature_map_cls = feature_map[i].squeeze(0)
        feature_map_cls = feature_map_cls[mask_.repeat(feature_map_cls.shape[0], 1, 1)]
        feature_map_cls = feature_map_cls.reshape(feature_map.shape[1],
                                                  int(feature_map_cls.shape[0] / feature_map.shape[1]))
        feature_map_cls = feature_map_cls.transpose(1, 0).unsqueeze(0)