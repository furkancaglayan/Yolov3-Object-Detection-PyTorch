from progress.bar import IncrementalBar as b


def create_progressbar(max_amount):
    return b('Processing', max=max_amount)

