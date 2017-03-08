which_config = 0


class Config0(object):
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 1
    num_steps = 1
    hidden_size = 40 # fix
    max_epoch = 14
    max_max_epoch = 1000
    keep_prob = 1
    lr_decay = 1 / 1.02
    batch_size = 100
    output_size = 40
    use_gpu = '/gpu:0'


class Config1(object):
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 1
    num_steps = 10
    hidden_size = 40  # fix
    max_epoch = 14
    max_max_epoch = 1000
    keep_prob = 1
    lr_decay = 1 / 1.02
    batch_size = 100
    output_size = 40
    use_gpu = '/gpu:1'


class Config2(object):
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 1
    num_steps = 10
    hidden_size = 40  # fix
    max_epoch = 14
    max_max_epoch = 1000
    keep_prob = 1
    lr_decay = 1 / 1.02
    batch_size = 100
    output_size = 40
    use_gpu = '/gpu:2'


class Default(object):
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 1
    num_steps = 10
    hidden_size = 40  # fix
    max_epoch = 14
    max_max_epoch = 1000
    keep_prob = 1
    lr_decay = 1 / 1.01
    batch_size = 100
    output_size = 40
    use_gpu = '/gpu:2'
