#include <stddef.h>
#include <stdio.h>

struct ops {
	int (*open)(void * self, int flags);
	int (*close)(void * self);
};

struct device_ops {
	int (*init)(void);
};

struct device {
	struct ops const * api;
	void * context;
	struct device_ops ops;
};

struct embedded_holder {
	struct {
		int x;
		int (*fn)(void);
	} inner;
};

struct handler_holder {
	void (*run)(void);
};

struct node {
	struct node * next;
	int (*fn)(void);
};

struct container {
	struct ops * looks_like_ops;
	int * ip;
};

typedef struct {
	int x;
} anon_t;

struct anon_wrapper {
	anon_t *anon;
};

union un {
	int a;
	float b;
};

struct bitpacked {
	unsigned a: 1;
	unsigned b: 3;
	unsigned: 2;
};

enum flags {
	FLAG_ZERO,
	FLAG_ONE,
};

typedef int ret_t;

static int driver_a_open(void * self, int flags);
static int driver_a_close(void * self);
static int driver_b_open(void * self, int flags);
static int driver_b_close(void * self);
static int node_fn(void);
static int enum_target(enum flags value);
static int dev_init(void);
static int anon_fn(void);

void plain_target(int value);
ret_t typedef_target(void);
void undef_ptr_target(void);

struct ops const ops_a = {.open = driver_a_open, .close = driver_a_close};
struct ops const ops_b = {.open = driver_b_open, .close = driver_b_close};
struct device dev_a = {.api = &ops_a, .context = 0, .ops = {.init = dev_init}};
struct device dev_b = {.api = &ops_b, .context = 0, .ops = {.init = dev_init}};
struct device dev_c = {.api = &ops_b, .context = (void *) &ops_a, .ops = {.init = dev_init}};
struct handler_holder holder = {.run = undef_ptr_target};
struct handler_holder bss_holder;
struct node node_a = {.next = &node_a, .fn = node_fn};
struct container wrong_chain = {.looks_like_ops = (struct ops *) &dev_a, .ip = 0};
struct container null_chain = {.looks_like_ops = 0, .ip = 0};
anon_t anon_obj = {.x = 1};
struct anon_wrapper wrap = {.anon = &anon_obj};
union un un_obj = {.a = 1};
struct bitpacked bits = {.a = 1, .b = 2};
struct embedded_holder holder2 = {.inner = {.x = 1, .fn = anon_fn}};

void (*plain_cb)(int) = plain_target;
void (*bss_cb)(int);
volatile int volatile_count;

static int const folded = 42;

int use_folded(void)
{
	return folded;
}

int main(void)
{
	int result = 0;

	plain_cb(1);
	if (bss_cb != 0) {
		bss_cb(2);
	}
	result += dev_a.api->open(dev_a.context, 2);
	result += dev_b.api->close(dev_b.context);
	holder.run();
	result += node_a.fn();
	result += wrap.anon->x;
	result += un_obj.a;
	result += bits.a;
	result += volatile_count;
	result += enum_target(FLAG_ONE);
	result += typedef_target();
	puts("device model");
	return result + use_folded();
}

void plain_target(int value)
{
	volatile_count = value;
}

ret_t typedef_target(void)
{
	return FLAG_ONE;
}

void undef_ptr_target(void)
{}

static int driver_a_open(void * self, int flags)
{
	return flags + (self != 0);
}

static int driver_a_close(void * self)
{
	return self != 0;
}

static int driver_b_open(void * self, int flags)
{
	return (self != 0) - flags;
}

static int driver_b_close(void * self)
{
	return -(self != 0);
}

static int node_fn(void)
{
	return 7;
}

static int dev_init(void)
{
	return 9;
}

static int anon_fn(void)
{
	return 3;
}

static int enum_target(enum flags value)
{
	return value;
}
