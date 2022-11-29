/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './templates/*.html',
    './*.py'
  ],
  theme: {
    extend: {
      backgroundImage: {
        'default': "url('/static/img/sky_bg.jpg')",
      },
      fontFamily: {
        baloo: [
          'Baloo 2',
          'cursive'
        ]
      },
      fontSize: {
      },
    },
    colors: {
      blue_txtbox: '#E2F1FF',
      blue_btn: '#3EB3F5',
      blue_hover: '#229DE2',
      blue_bg: '#9CDEF6',
      blue_bg2: '#6dcbed',
      gold: '#FFD700',
      orange: '#FFA800',
      gray: '#363636',
      inputbox: '#edf8fa',
      white: '#FFFFFF',
    },
    fontWeight: {
      regular: 400,
      medium: 500,
      semibold: 600,
    },
    borderRadius: {
      'large': '20px'
    }
  },
  plugins: [],
}
