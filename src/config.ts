export const SITE = {
  website: "https://lovelonelytime.netlify.app/",
  author: "Honghao Jia",
  profile: "https://github.com/LoveLonelyTime",
  desc: "LoveLonelyTime's Blog",
  title: "LoveLonelyTime's Blog",
  ogImage: "lovelonelytime-og.webp", // located in public
  lightAndDarkMode: true,
  postPerIndex: 6,
  postPerPage: 8,
  scheduledPostMargin: 15 * 60 * 1000, // 15 minutes
  showArchives: true,
  showGalleries: false,
  showBackButton: false, // show back button in post detail
  editPost: {
    enabled: true,
    text: "Edit",
    url: "https://github.com/LoveLonelyTime/blog/edit/main/",
  },
  dynamicOgImage: false,
  dir: "ltr", // "rtl" | "auto"
  lang: "zh", // html lang code. Set this empty and default will be "en"
  timezone: "Asia/Shanghai", // Default global timezone (IANA format) https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
  introAudio: {
    enabled: false, // mostrar/ocultar el reproductor en el hero
    src: "/audio/intro-web.mp3", // ruta al archivo (relativa a /public)
    label: "INTRO.MP3", // etiqueta display en el reproductor
    duration: 30, // duración en segundos (para la barra de progreso fija)
  },
} as const;
