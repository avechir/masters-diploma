def filter_low_confidence(rgb_results, threshold):
    rgb_lowconfidence = []
    lowconf_obj = set()
    for r in rgb_results:
        if r['confidence'] < threshold:
            lowconf_obj.add(r['file'])
    for r in rgb_results:
        if r['file'] in lowconf_obj:
            rgb_lowconfidence.append(r)

    return rgb_lowconfidence

def should_remove_object(obj, notanobjectlist):
    for not_obj in notanobjectlist:
        if (obj["file"] == not_obj["file"] and
            obj["timestampsec"] == not_obj["timestampsec"] and
            obj["timestampnsec"] == not_obj["timestampnsec"] and
            obj["bbox"] == not_obj["bbox"]):
            return True
    return False

def filter_results(rgb_results, notanobjectlist):
    filtered_rgb_results = []
    for obj in rgb_results:
        if not should_remove_object(obj, notanobjectlist):
            filtered_rgb_results.append(obj)
    return filtered_rgb_results