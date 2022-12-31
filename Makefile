dist:
	cd frontend && pnpm i && pnpm build && cd .. && cp -r frontend/dist ./
init-submodule:
	git submodule update --init --recursive

all: init-submodule dist