void handler(void) {}

void ( * hook)(void);

int volatile handled;

[[gnu::noipa]] void dispatch(void) {
	unsigned char volatile frame[24];
	frame[0] = 1;
	hook();
	handled = frame[0];
}

[[gnu::noipa]] void forward(void) {
	hook();
}

int main(void) {
	hook = handler;
	dispatch();
	forward();
	return handled;
}
