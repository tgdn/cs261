import App from './components/app'
import HomePage from './components/pages/home'

export default [
    {
        path: '/',
        component: App,
        indexRoute: {
            component: HomePage,
            // onEnter:(data, replace) => {
            //     replace('/members/dashboard')
            // }
        },
        childRoutes: [
            {
                path: 'random-route',
                component: null,
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
        //     }, {
        //         path: '*',
        //         component: null
        //     }
        ]
    }
]
