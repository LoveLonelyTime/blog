import type { Props } from "astro";
import IconMail from "@/assets/icons/IconMail.svg";
import IconGitHub from "@/assets/icons/IconGitHub.svg";
import IconBilibili from "@/assets/icons/IconBilibili.svg";
import IconBrandX from "@/assets/icons/IconBrandX.svg";
import IconCV from "@/assets/icons/IconCV.svg";
import { SITE } from "@/config";

interface Social {
  name: string;
  href: string;
  linkTitle: string;
  icon: (_props: Props) => Element;
}

export const SOCIALS: Social[] = [
  {
    name: "GitHub",
    href: "https://github.com/LoveLonelyTime/",
    linkTitle: `${SITE.author} on GitHub`,
    icon: IconGitHub,
  },
  {
    name: "Bilibili",
    href: "https://space.bilibili.com/23011999",
    linkTitle: `${SITE.author} on Bilibili`,
    icon: IconBilibili,
  },
  {
    name: "Mail",
    href: "mailto:jiahonghao2002@gmail.com",
    linkTitle: `Mail to ${SITE.author}`,
    icon: IconMail,
  },
  {
    name: "CV",
    href: "/docs/CV.pdf",
    linkTitle: `${SITE.author}'s CV`,
    icon: IconCV,
  },
] as const;

export const SHARE_LINKS: Social[] = [
  {
    name: "X",
    href: "https://x.com/intent/post?url=",
    linkTitle: `Share on X`,
    icon: IconBrandX,
  },
  {
    name: "Mail",
    href: "mailto:?subject=See%20this%20post&body=",
    linkTitle: `Share by mail`,
    icon: IconMail,
  },
] as const;
