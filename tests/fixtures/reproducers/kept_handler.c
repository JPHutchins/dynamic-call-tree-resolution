extern void reset(void);

void ( * volatile restart)(void) = reset;

int main(void) {
	return restart == nullptr;
}
