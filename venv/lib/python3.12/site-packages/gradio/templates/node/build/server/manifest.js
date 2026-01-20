const manifest = (() => {
function __memo(fn) {
	let value;
	return () => value ??= (value = fn());
}

return {
	appDir: "_app",
	appPath: "_app",
	assets: new Set([]),
	mimeTypes: {},
	_: {
		client: {start:"_app/immutable/entry/start.JDUSfjgp.js",app:"_app/immutable/entry/app.BVHP6c21.js",imports:["_app/immutable/entry/start.JDUSfjgp.js","_app/immutable/chunks/CCr6UdYm.js","_app/immutable/entry/app.BVHP6c21.js","_app/immutable/chunks/PPVm8Dsz.js"],stylesheets:[],fonts:[],uses_env_dynamic_public:false},
		nodes: [
			__memo(() => import('./chunks/0-tdAi3kFl.js')),
			__memo(() => import('./chunks/1-CTgTJai2.js')),
			__memo(() => import('./chunks/2-B_-s2cNJ.js'))
		],
		remotes: {
			
		},
		routes: [
			{
				id: "/[...catchall]",
				pattern: /^(?:\/([^]*))?\/?$/,
				params: [{"name":"catchall","optional":false,"rest":true,"chained":true}],
				page: { layouts: [0,], errors: [1,], leaf: 2 },
				endpoint: null
			}
		],
		prerendered_routes: new Set([]),
		matchers: async () => {
			
			return {  };
		},
		server_assets: {}
	}
}
})();

const prerendered = new Set([]);

const base = "";

export { base, manifest, prerendered };
//# sourceMappingURL=manifest.js.map
