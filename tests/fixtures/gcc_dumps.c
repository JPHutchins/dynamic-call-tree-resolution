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

struct driver {
	struct api ops[2];
	int (*handlers[2])(int);
};

extern void escape(void * object);

int through_array_member(struct driver * driver, int i, int x) {
	return driver->ops[i].run(x);
}

void array_member_store(struct driver * driver, int i) {
	driver->ops[i].run = put_char;
}

int through_member_table(struct driver * driver, int i, int x) {
	return driver->handlers[i](x);
}

int through_local_array(int i, int j, int x) {
	struct api grid[2][3];
	escape(grid);
	grid[i][j].run = put_char;
	escape(grid);
	return grid[i][j].run(x);
}

int through_address_taken(int (*fn)(int), int x) {
	escape(&fn);
	return fn(x);
}

int through_local_static(int x, int (*next)(int)) {
	static int (*saved)(int);
	int ( * const previous)(int) = saved;
	saved = next;
	return previous(x);
}

void static_record_store(void) {
	static struct api local_api;
	local_api.run = put_char;
	escape(&local_api);
}

int through_array_pointer(struct api(*apis)[3], int i, int x) {
	return (*apis)[i].run(x);
}

struct api api_0;
int (*hook_1)(int);
extern int (*pick(int))(int);
extern int dev_1(int);

void suffixed_stores(void) {
	api_0.run = put_char;
	hook_1 = put_char;
}

int through_suffixed(int x) {
	return api_0.run(x) + hook_1(x);
}

void call_stores(struct api * api) {
	api->run = pick(1);
	hook_1 = pick(2);
}

int through_direct_suffix(int (*dev)(int), int x) {
	return dev_1(x) + dev(x);
}

struct big {
	int words[4];
};

typedef struct big(*big_maker)(int);

void through_return_slot(big_maker make) {
	struct big made = make(1);
	escape(&made);
}

struct big through_returned_slot(big_maker make) {
	return make(2);
}

void through_stored_result(big_maker make, void * buffer) {
	*(struct big *) buffer = make(3);
}
