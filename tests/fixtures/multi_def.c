struct ops {
	int (*open)(void *self, int flags);
	int (*close)(void *self);
};

struct device {
	const struct ops *api;
	void *context;
};

static int hidden_open(void *self, int flags)
{
	return flags;
}

static int hidden_close(void *self)
{
	return 0;
}

const struct ops ops_hidden = { .open = hidden_open, .close = hidden_close };
struct device dev_x = { .api = &ops_hidden, .context = 0 };
