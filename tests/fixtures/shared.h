struct plain {
	int (*run)(void);
};

struct shared {
	int (*fn)(void);
#ifdef SECOND
	int (*extra)(void);
#endif
};
