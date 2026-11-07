struct api {
	int (*run)(int);
};

struct device {
	struct api const * api;
};

typedef struct stream {
	int (*put)(int);
} stream_t;

struct holder {
	struct {
		int (*fn)(int);
	} inner;
	union {
		int (*first)(int);
		long (*second)(long);
	};
};

struct boot {
	void ( * volatile fn)(void);
};

static int (*hook)(int);
int (*table[2])(int);
struct boot boot_params;

int put_char(int c) {
	return c;
}

__attribute__((noinline)) int counter_4(int x) {
	return x + 1;
}

int through_field(struct device const * dev, int x) {
	return dev->api->run(x);
}

int through_parameter(int (*cb)(int), int x) {
	return cb(x);
}

int through_variable(int x) {
	return hook(x);
}

int through_element(int i, int x) {
	return table[i](x);
}

int through_nested(struct holder * holder, int x) {
	return holder->inner.fn(x);
}

int through_anonymous(struct holder * holder, int x) {
	return holder->first(x);
}

void through_volatile(void) {
	boot_params.fn();
}

int direct(int x) {
	return counter_4(x);
}

void install(int (*fn)(int)) {
	hook = fn;
}

int typedef_store(stream_t * s, int x) {
	s->put = put_char;
	return s->put(x);
}

void field_stores(struct holder * holder, int (*fn)(int)) {
	holder->inner.fn = fn;
	holder->first = 0;
}

struct api apis[3];
extern struct api more_apis[];
int counter;

int through_cast(unsigned long address, int x) {
	return ((int (*)(int)) address) (x);
}

int through_deref(int (**pointer)(int), int x) {
	return (*pointer)(x);
}

int through_struct_element(int i, int x) {
	return apis[i].run(x);
}

int through_unsized_element(int i, int x) {
	return more_apis[i].run(x);
}

void data_stores(struct holder * holder) {
	counter = 1;
	holder->inner.fn = put_char;
}

typedef int count_t;

count_t counted;
