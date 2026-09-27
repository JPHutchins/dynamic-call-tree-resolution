struct ops {
	int (*open)(void * self, int flags);
	int (*close)(void * self);
};

struct device {
	struct ops const * api;
	void * context;
};

static int hidden_open(void * self, int flags)
{
	(void) self;
	return flags;
}

static int hidden_close(void * self)
{
	(void) self;
	return 0;
}

struct ops const ops_hidden = {.open = hidden_open, .close = hidden_close};
struct device dev_x = {.api = &ops_hidden, .context = 0};
