import App from './components/app'
import HomePage from './components/pages/home'
import SymbolPage from './components/pages/symbol'
import PageDoesNotExist from './components/pages/404'

export default [
    {
        path: '/',
        component: App,
        indexRoute: {
            component: HomePage,
        },
        childRoutes: [
            {
                path: ':symbol',
                component: SymbolPage,
            },
            {
                path: '*',
                component: PageDoesNotExist,
            }
        //     {
        //         path: 'dashboard',
        //         component: null,
        //         indexRoute: {
        //             component: null
        //         },
        //         childRoutes: [
        //             {
        //                 path: 'events',
        //                 component: null
        //             },
        //             {
        //                 path: 'billing',
        //                 component: null
        //             }
        //         ]
        //     },
        ]
    }
]
