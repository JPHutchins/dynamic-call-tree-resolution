struct ops {
	int (*open)(void * self, int flags);
	int (*close)(void * self);
};

struct device {
	struct ops const * api;
	void * context;
};

extern struct device dev_x;

int main(void) {
	return dev_x.api->open(dev_x.context, 1);
}
