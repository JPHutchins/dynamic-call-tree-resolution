typedef struct {
	int (*run)(void);
} anon_ops_t;

struct anon_ops_holder {
	anon_ops_t const * ops;
};

static int anon_ops_fn(void);

anon_ops_t const anon_ops = {.run = anon_ops_fn};
struct anon_ops_holder const anon_ops_holder = {.ops = &anon_ops};

struct {
	int (*run)(void);
} const bare_anon = {.run = anon_ops_fn};

anon_ops_t dynamic_anon_ops;

struct {
	int x;
} anonymous_state;

struct visitor {
	void (*visit_state)(__typeof__(anonymous_state) *);
	void (*visit_ops)(anon_ops_t const *);
};

struct visitor visitor;

static int anon_ops_fn(void) {
	return 4;
}

int main(void) {
	return (
		anon_ops_holder.ops->run() +
		bare_anon.run() +
		(dynamic_anon_ops.run == nullptr) +
		(visitor.visit_ops == nullptr)
	);
}
