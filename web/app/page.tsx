import { redirect } from 'next/navigation'

export default function HomePage() {
  // Redirect to the predict page as the default
  redirect('/predict')
}
